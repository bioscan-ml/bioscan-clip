#!/usr/bin/env python3

import argparse
import csv
import copy
import math
import os
import statistics
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from collections import Counter
from collections import namedtuple

NamedCounts = namedtuple('NamedCounts', ['name', 'counts', 'colors'])

# read csv with split, level, label, num_records (number of records for each class label)
def read_label_counts(filename):
    counts = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            split = row['split']
            level = row['level']
            if split not in counts:
                counts[split] = {}
            if level not in counts[split]:
                counts[split][level] = Counter()
            counts[split][level].update({ row['label']: int(row['num_records'])})
    return counts

# write csv with number of records for each label (for different splits and taxonomic levels)
def write_label_counts(filename, counts):
    with open(filename, 'w') as output:
        header = ['split','level','label','num_records']
        print(','.join(header), file=output)
        for split,split_counts in counts.items():
            for level,c in split_counts.items():
                for k,v in c.items():   
                    print(','.join([split, level, k, str(v)]), file=output)

# read csv with split, level, num_records, count (count is number of labels with that many number of records)
def read_level_counts(filename):
    counts = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            split = row['split']
            level = row['level']
            if split not in counts:
                counts[split] = {}
            if level not in counts[split]:
                counts[split][level] = Counter()
            counts[split][level].update({ int(row['num_records']): int(row['count'])})
    return counts

# write csv with distribution of number of classes with given number of records (for different splits and taxonomic levels)
def write_level_counts(filename, counts):
    with open(filename, 'w') as output:
        header = ['split','level','num_records','count']
        print(','.join(header), file=output)
        for split,split_counts in counts.items():
            for level,c in split_counts.items():
                for k,v in c.items():   
                    print(','.join([split, level, str(k), str(v)]), file=output)

# for csv of samples (each sample has the label for each taxonomic level), 
# count how many records there are for each label (by split, level)
def count_frequencies(input, levels):
    counts = {}
    with open(input, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            split = row['split']
            if split not in counts:
                counts[split] = {}
                for level in levels:
                    counts[split][level] = Counter()
            for level in levels:
                counts[split][level].update([row[level]])
    return counts

# group label counts together using mapping 
def get_grouped_split_frequencies(counts, mapping):
    mapped_counts = {}
    for raw_split,split_counts in counts.items():
        split = mapping[raw_split]
        if split not in mapped_counts:
            mapped_counts[split] = {}
            for level in split_counts.keys():
                mapped_counts[split][level] = Counter()
        for level,c in split_counts.items():
            mapped_counts[split][level].update(c)
    return mapped_counts

# take label counts, and get distribution of number of records to number of labels
def get_dist(counts):
    dist = {}
    for split,split_counts in counts.items():
        dist[split] = {}
        for level,c in split_counts.items():
            dist[split][level] = Counter()
            for k,v in c.items():
                if k != 'not_classified':
                    dist[split][level].update([v])
    return dist

# take a set of ranges and expand them based on step size
def expand_ranges(ranges):
    full_ranges = []
    for i,r in enumerate(ranges):
        if 'step' in r:
            r['expanded_ranges'] = []
            for x in range(r['min'], r['max'] + 1, r['step']):
                r2 = {'min': x, 'max': x + r['step'] - 1, 'expanded': i }
                r['expanded_ranges'].append(r2)
                full_ranges.append(r2)
        else:
            r2 = {k:v for k,v in r.items()}
            r['expanded_ranges'] = [r2]
            r2['expanded'] = None      
            full_ranges.append(r2)
    for i,r in enumerate(full_ranges):
        r['index'] = i
    return full_ranges

# populate range label
def populate_range_labels(ranges):
    for range_def in ranges:
        if range_def['min'] == range_def['max']:
            range_label = f"{range_def['min']}"
        elif range_def['max'] is None or math.isinf(range_def['max']):
            range_label = f"{range_def['min']}-max"
            range_def['max'] = float('inf')
        else:
            range_label = f"{range_def['min']}-{range_def['max']}"
        range_def['label'] = range_label

# Handler for legend with multiple colors
class MultiColors:
    def __init__(self, colors, label=None):
        self.colors = colors
        self.label = label

    def get_label(self):
        return self.label


class MultiColorsHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        patches = self.create_artists(legend, orig_handle,
                                      handlebox.xdescent, handlebox.ydescent,
                                      handlebox.width, handlebox.height,
                                      fontsize, 
                                      handlebox.get_transform())
        return PatchCollection(patches)

    def create_artists(self, legend, orig_handle,
                   xdescent, ydescent, width, height, fontsize,
                   trans):
        colors = orig_handle.colors
        x0, y0 = -xdescent, ydescent            # not sure why x0 is negative of xdescent
        width, height = width / len(colors), height
        patches = []
        for i,color in enumerate(colors):
            patch = Rectangle([x0 + i*width, y0], width, height, facecolor=color,
                              edgecolor=color, 
                              transform = trans)
            patches.append(patch)
        return patches
        

# Plot distribution of labels to number of records as color bar
def plot_count_in_splits_as_colors(rows, ranges, expand=False, color_tags=None, 
                                   xlabel='Number of species', title='Distribution of species',
                                   filename=None, hide_legend=False):
    # make copy
    ranges = copy.deepcopy(ranges)
    for i,r in enumerate(ranges):
        r['index'] = i

    # get ranges
    if expand:
        populate_range_labels(ranges)
        full_ranges = expand_ranges(ranges)
    else:
        full_ranges = ranges

    populate_range_labels(full_ranges)

    # colors
    colors = {
        'blue': ['#B7CCE7FF', '#499DD8FF', '#1F77B4FF', '#01497CFF', '#150150FF'],
        'orange': ['#F8DFC5FF', '#F6B371FF', '#FA9236FF', '#D86702FF', '#602400FF'],
        'green': ['#C8F6BFFF', '#98df8a', '#6CC15BFF', '#005500FF', '#051900FF'] 
    }

    color_bins = {}
    color_max = {}
    for k,cs in colors.items():
        color_bins[k] = []
        color_max[k] = 0

    # Initialize data structure
    data = {}
    label_info = {}
    for ri,row in enumerate(rows):
        for freq_list,color_name in zip(row.counts, row.colors):
            color_max[color_name] = max(color_max[color_name], max(freq_list.keys())) 
            for i,range_def in enumerate(full_ranges):
                count = sum([c for nr,c in freq_list.items() if range_def['min'] <= nr <= range_def['max']])
                label = f'range_{i}-{color_name}'
                if label not in data:
                    data[label] = [0] * len(rows)
                data[label][ri] = count    
                if label not in label_info:
                    label_info[label] = { 'color_name': color_name, 'range': range_def }
                if i not in color_bins[color_name]:
                    color_bins[color_name].append(i)

    cmaps = {}
    expanded_orig_ranges = [i for i,r in enumerate(ranges) if len(r.get('expanded_ranges',[])) > 1]
    for color_name,bins in color_bins.items():
        sorted_bins = sorted(bins)
        expanded_count = 0
        if expand:
            for i in sorted_bins:            
                label = f'range_{i}-{color_name}'
                li = label_info[label]
                if li['range']['expanded'] is not None:
                    expanded_count += 1
        total = len(sorted_bins)

        cs = colors[color_name]
        if expand and len(expanded_orig_ranges):
            expanded_count1 = len(ranges[expanded_orig_ranges[0]]['expanded_ranges'])
            cmap1 = LinearSegmentedColormap.from_list("custom_cmap1", [cs[0], cs[1]], N=expanded_count1)
            cmap2 = LinearSegmentedColormap.from_list("custom_cmap2", [cs[2], cs[-1]], N=(total-expanded_count1))
            cmap = ListedColormap([cmap1(i) for i in range(cmap1.N)] + [cmap2(i) for i in range(cmap2.N)])
        else:
            cmap = LinearSegmentedColormap.from_list("custom_cmap", [cs[0], cs[-1]], N=total)
        cmaps[color_name] = cmap
        for i in sorted_bins:            
            label = f'range_{i}-{color_name}'
            li = label_info[label]
            color = cmap(i / (total - 1)) 
            li['color'] = color
            range_label = li['range']['label']
            range_label = range_label.replace('max', f'{color_max[color_name]}')
            if color_tags is not None:
                li['label'] = range_label + ' (' + color_tags[color_name] + ')'
            else:
                li['label'] = range_label

    # Data preparation
    row_names = [r.name for r in rows]

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.invert_yaxis()
    bottom = np.zeros(len(row_names))
    for i, (label, counts) in enumerate(data.items()):
        li = label_info[label]
        range_label = li['label']
        ax.barh(row_names, counts, left=bottom, color=li['color'], label=range_label)
        bottom += np.array(counts)
    if not hide_legend:
        ax.set_xlabel(xlabel, fontsize=32)
    ax.set_title(title, fontsize=32)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)

    legend_handler_map = {
                   MultiColors: MultiColorsHandler(), 
                   tuple: HandlerTuple(ndivide=None,pad=0.0)}
    # labels for colors
    if color_tags is not None:
        legend_elements = []
        for color_name in color_tags:
            color = colors[color_name][1]
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label=color_tags[color_name]))
            #cmap = cmaps[color_name]
            #legend_elements.append(MultiColors([cmap(i) for i in range(cmap.N)], label=color_tags[color_name]))
        if not hide_legend:
            color_legend = plt.legend(handles=legend_elements,
                                      handler_map=legend_handler_map,
                                      loc='upper center',
                                      bbox_to_anchor=(1.0, -0.15),
                                      fontsize=24, ncols=len(color_tags),
                                      handletextpad=0.4,
                                      columnspacing=1.0,
                                      borderpad=0.2,
                                      frameon=True)
            ax.add_artist(color_legend)

    # create custom legend elements
    legend_elements = []
    legend_labels = []
    max_value = max(color_max.values())
    for i,range_def in enumerate(ranges):
        els = []
        for color_name,bins in color_bins.items():
            if len(bins) > 0:
                expanded_ranges = range_def.get('expanded_ranges', [range_def])
                if len(expanded_ranges) > 1:
                    expanded_bin_indices = [r['index'] for r in expanded_ranges]
                    cmap = cmaps[color_name]
                    els.append(MultiColors([cmap(i) for i in expanded_bin_indices]))
                else:
                    j = expanded_ranges[0]['index']
                    label = f'range_{j}-{color_name}'
                    li = label_info[label]
                    color = li['color']
                    range_label = li['label']
                    els.append(Patch(facecolor=color, edgecolor=color, label=range_label))
        if len(els) > 0:
            legend_elements.append(tuple(els))
            range_label = range_def['label']
            range_label = range_label.replace('max', f'{max_value}')
            legend_labels.append(range_label)
    if not hide_legend:
        plt.legend(legend_elements, legend_labels,
                   handler_map=legend_handler_map,
                   handletextpad=0.4,
                   labelspacing=0.3,
                   loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)

    plt.tight_layout()
    if hide_legend:
        fig.subplots_adjust(left=0.14, right=0.83, wspace=0, hspace=0)
    else:
        fig.subplots_adjust(left=0.14, wspace=0, hspace=0)
    print('Saving to', filename)
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()



# Plot distribution of labels to number of records as histogram
def plot_count_in_splits_as_histogram(rows, ranges, expand=False, color_tags=None, 
                                      ylabel=None, xlabel='Number of species', title='Distribution of species',
                                      filename=None):
    # get ranges
    if expand:
        full_ranges = expand_ranges(ranges)
    else:
        full_ranges = ranges

    populate_range_labels(full_ranges)

    # colors
    colors = {
        'blue': '#499DD8FF',
        'orange': '#F6B371FF',
        'green': '#2ca02c'
    }

    max_num_records = 0
    max_count = 0

    # Initialize data structure
    data = []
    for ri,row in enumerate(rows):
        data_element = {}
        for freq_list,color_name in zip(row.counts, row.colors):
            max_num_records = max(max_num_records, max(freq_list.keys())) 
            for i,range_def in enumerate(full_ranges):
                count = sum([c for nr,c in freq_list.items() if range_def['min'] <= nr <= range_def['max']])
                max_count = max(max_count, count)
                if color_name not in data_element:
                    data_element[color_name] = []
                data_element[color_name].append(count)
        data.append(data_element)

    for range_def in full_ranges:
        range_def['label'] = range_def['label'].replace('max', f'{max_num_records}')
    range_labels = [r['label'] for r in full_ranges]

    # Data preparation
    row_names = [r.name for r in rows]

    # Plotting
    fig, ax = plt.subplots(len(row_names), figsize=(18, 4))

    grouped = True
    if grouped:
        # grouped bar chart
        max_group_count = max([len(r.counts) for r in rows])
        for ri,data_element in enumerate(data):
            width = 0.4  # the width of the bars
            x = np.arange(len(range_labels))  # the label locations
            group_count = len(rows[ri].counts) 
            multiplier = 0
            for i, (label, counts) in enumerate(data_element.items()):
                offset = width * multiplier
                rects = ax[ri].bar(x + offset, counts, width, color=colors[label], label=label)
                bar_labels = [v if v > 0 else "" for v in rects.datavalues] 
                ax[ri].bar_label(rects, padding=1, labels=bar_labels, color=colors[label])
                multiplier += 1
            ax[ri].set_xticks(x + (group_count - 1) * width/2, range_labels)
            if ri < len(row_names) - 1:
                ax[ri].set_xticklabels([])
            ymax = math.floor(max_count*1.3)    
            ax[ri].set_ylim([0, ymax])
            ax[ri].text(len(range_labels) * width, max_count*0.9, row_names[ri], fontsize=14, horizontalalignment='center')
            #ax[ri].set_title(row_names[ri], fontsize=20)

    else:
        # stacked bar chart
        for ri,data_element in enumerate(data):
            bottom = np.zeros(len(range_labels))
            for i, (label, counts) in enumerate(data_element.items()):
                ax[ri].bar(range_labels, counts, color=colors[label], label=label, bottom=bottom)
                bottom += np.array(counts)
            if ri < len(row_names) - 1:
                ax[ri].set_xticklabels([])
            #ax[ri].set_title(row_names[ri], fontsize=20)

    if ylabel is not None:
        ri = math.floor(len(row_names) / 2)
        ax[ri].set_ylabel(ylabel, fontsize=20)
    ax[-1].set_xlabel(xlabel, fontsize=20)
    ax[0].set_title(title, fontsize=32)

    if color_tags is not None:
        legend_elements = []
        for color_name in color_tags:
            color = colors[color_name]
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label=color_tags[color_name]))
        color_legend = plt.legend(handles=legend_elements, loc='upper right', 
                                  bbox_to_anchor=(1.0, -0.15),
                                  fontsize=20, ncols=len(color_tags),
                                  handletextpad=0.4,
                                  columnspacing=1.0,
                                  borderpad=0.2,
                                  frameon=True)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    print('Saving to', filename)
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

# Latex table stubs

def print_table(rows, title, has_header=True, output=sys.stdout):
    print(title, file=output)
    for i,row in enumerate(rows):        
        print(' & '.join(row) + ' \\\\', file=output)
        if has_header and i == 0:
            print('\\midrule', file=output)

def get_average_num_records(counts, levels):
    splits = ['train', 'seen_keys', 'val_seen_query', 'test_seen_query', 'val_unseen_keys', 'val_unseen_query', 'test_unseen_keys', 'test_unseen_query']
    rows = [splits]
    for level in levels:
        ms = [level]
        for split in splits:
            counter = counts[split][level]            
            total_taxa = sum(counter.values())
            total_records = sum(counter.keys())
            #minimum = min(counter.keys())
            #maximum = max(counter.keys())
            m = total_records / total_taxa
            ms.append("{:.2f}".format(m))
            #ms.append("{:.2f}".format(minimum))
            #ms.append("{:.2f}".format(maximum))
        rows.append(ms)
    return rows


def print_average_num_records(counts, level, output=sys.stdout):
    rows = get_average_num_records(counts, level)
    print_table(rows, '% Average number records (copy into your overleaf table)', has_header=True, output=output)


def get_overlap_statistics(seenunseen_counts, levels):
    header = ['', 'total', 'seen', 'unseen', 'seen', 'unseen', 'seen', 'unseen', 'overlap']
    rows = [header]
    for level in levels:
        level_labels = {}
        splits = ['none', 'single', 'seen', 'unseen']
        for s in splits:
            counter = seenunseen_counts[s][level]
            level_labels[s] = set([k for k in counter.keys() if k != 'not_classified'])
        all_labels = level_labels['seen'] | level_labels['unseen'] | level_labels['none']

        ms = [level]
        pairs = [('none', 'seen'), ('none', 'unseen'),
                 ('single', 'seen'), ('single', 'unseen'),
                 ('seen', 'seen'), ('unseen', 'unseen'), ('seen', 'unseen')]
        for s1,s2 in pairs:
            labels1 = level_labels[s1]
            all_labels.update(labels1)
            labels2 = level_labels[s2]
            overlap = len(labels1 & labels2)
            ms.append(str(overlap))
        ms.insert(1, str(len(all_labels)))
        rows.append(ms)
    return rows

def hmean(a, b):
    return statistics.harmonic_mean([a,b])


def get_chance_accuracies(valtest_seenunseen_counts, query_key_counts, levels):
    header = ['', 'val/test seen', 'val unseen', 'test unseen', 'val H.M.', 'test H.M.', 
                  'seen', 'val unseen', 'test unseen', 'val H.M.', 'test H.M.']
    rows = [header]
    for level in levels:
        level_labels = {}
        splits = ['val_seen', 'test_seen', 'val_unseen', 'test_unseen']
        for s in splits:
            counter = valtest_seenunseen_counts[s][level]
            level_labels[s] = set([k for k in counter.keys() if k != 'not_classified'])
        level_labels['seen'] = level_labels['val_seen'] | level_labels['test_seen']
        level_labels['unseen'] = level_labels['val_unseen'] | level_labels['test_unseen']

        key_sets = ['seen_keys', 'val_unseen_keys', 'test_unseen_keys']
        most_freq_label_counts = []
        for key in key_sets:
            counter = query_key_counts[key][level]
            most_freq_label_counts.append(counter.most_common(1)[0])
        split_to_keyindex = [0, 0, 1, 2]

        ms = []
        # (micro accuracy) - assuming most frequent class in key set        
        for si,s in enumerate(splits):
            label_count = most_freq_label_counts[split_to_keyindex[si]]
            counter = query_key_counts[f'{s}_query'][level]
            total = sum(counter.values())
            acc = label_count[1] / total
            #print(label_count, total, acc)
            ms.append(acc)

        ms.append(hmean(ms[0], ms[2]))
        ms.append(hmean(ms[1], ms[3]))

        # assumes equal chance of classifying to each class
        num_labels = [len(level_labels[k]) for k in ['seen', 'val_unseen', 'test_unseen']]
        for n in num_labels:
            ms.append(1/n)
        ms.append(hmean(ms[6], ms[8]))
        ms.append(hmean(ms[7], ms[8]))
        ms.pop(0)
        rows.append([level] + ["{:.2f}".format(100*v) for v in ms ])
    return rows


def print_overlap_statistics(seenunseen_counts, levels, output=sys.stdout):
    rows = get_overlap_statistics(seenunseen_counts, levels) 
    print_table(rows, '% Overlap statistics (copy into your overleaf table)', has_header=True, output=output)


def print_chance_accuracies(valtest_seenunseen_counts, query_key_counts, levels, output=sys.stdout):
    rows = get_chance_accuracies(valtest_seenunseen_counts, query_key_counts, levels) 
    print_table(rows, '% Chance accuracies (copy into your overleaf table)', has_header=True, output=output)

def print_overlap_and_chance(seenunseen_counts, valtest_seenunseen_counts, query_key_counts, levels, output=sys.stdout):
    rows1 = get_overlap_statistics(seenunseen_counts, levels) 
    rows2 = get_chance_accuracies(valtest_seenunseen_counts, query_key_counts, levels) 
    for r in rows2:
        r.pop(0)
    rows = [r1 + r2 for r1,r2 in zip(rows1,rows2)]
    print_table(rows, '% Overlap statistics (copy into your overleaf table)', has_header=True, output=output)



trainvaltest_split_map = {
    'no_split': 'train',
    'seen_keys': 'eval',
    'single_species': 'single',
    'test_seen': 'test',
    'test_unseen': 'test',
    'test_unseen_keys': 'test',
    'train_seen': 'train',
    'val_seen': 'val',
    'val_unseen': 'val',
    'val_unseen_keys': 'val'
}

querykey_split_map = {
    'no_split': 'train',
    'seen_keys': 'seen_keys',
    'single_species': 'single',
    'test_seen': 'test_seen_query',
    'test_unseen': 'test_unseen_query',
    'test_unseen_keys': 'test_unseen_keys',
    'train_seen': 'train',
    'val_seen': 'val_seen_query',
    'val_unseen': 'val_unseen_query',
    'val_unseen_keys': 'val_unseen_keys'
}

noneseenunseen_split_map = {
    'no_split': 'none',
    'seen_keys': 'seen',
    'single_species': 'single',
    'test_seen': 'seen',
    'test_unseen': 'unseen',
    'test_unseen_keys': 'unseen',
    'train_seen': 'seen',
    'val_seen': 'seen',
    'val_unseen': 'unseen',
    'val_unseen_keys': 'unseen'
}

valtest_seenunseen_split_map = {
    'no_split': 'none',
    'seen_keys': 'seen',
    'single_species': 'single',
    'test_seen': 'test_seen',
    'test_unseen': 'test_unseen',
    'test_unseen_keys': 'test_unseen',
    'train_seen': 'seen',
    'val_seen': 'val_seen',
    'val_unseen': 'val_unseen',
    'val_unseen_keys': 'val_unseen'
}


def main():
    parser = argparse.ArgumentParser(description="Create visualization of distribution of species count")
    parser.add_argument('-i', '--input', type=str, help='Input split file', default=f'data/BIOSCAN_1M/bioscan_1m_with_split.csv')
    parser.add_argument('-c', '--counts', type=str, help='Counts file', default=f'raw_split_counts.csv')
    args = parser.parse_args()

    levels = ['order', 'family', 'genus', 'species']
    if os.path.isfile(args.counts):
        label_counts = read_label_counts(args.counts)
    else:
        label_counts = count_frequencies(args.input, levels)
        write_label_counts(args.counts, label_counts)

    trainval_level_counts = get_grouped_split_frequencies(label_counts, trainvaltest_split_map)
    querykey_level_counts = get_grouped_split_frequencies(label_counts, querykey_split_map)
    noneseenunseen_level_counts = get_grouped_split_frequencies(label_counts, noneseenunseen_split_map)
    valtest_unseen_level_counts = get_grouped_split_frequencies(label_counts, valtest_seenunseen_split_map)
    print_overlap_statistics(noneseenunseen_level_counts, levels)
    print_chance_accuracies(valtest_unseen_level_counts, querykey_level_counts, levels)

    # Define range list
    ranges_colorbar = [{'min': 1, 'max': 10, 'step': 1 }, 
              {'min': 11, 'max': 20}, {'min': 21, 'max': 40}, {'min': 41, 'max': 80},
              {'min': 81, 'max': 160}, {'min': 161, 'max': None }]
    ranges = [{'min': 1, 'max': 10, 'step': 1 }, 
              {'min': 11, 'max': 20, 'step': 5}, {'min': 21, 'max': 40}, {'min': 41, 'max': 80},
              {'min': 81, 'max': 160}, {'min': 161, 'max': None }]

    counts = get_dist(querykey_level_counts)
    print_average_num_records(counts, levels)

    plot_folder = 'output_plots'
    os.makedirs(plot_folder, exist_ok=True)
    for level in levels:
        plot_count_in_splits_as_colors( [ NamedCounts('Train', [ counts['train'][level]], ['blue']),
                                NamedCounts('Val', [ counts['val_seen_query'][level], counts['val_unseen_query'][level] ], ['blue', 'orange']),
                                NamedCounts('Test', [ counts['test_seen_query'][level], counts['test_unseen_query'][level] ], ['blue', 'orange']) ],
                                ranges, expand=True,
                                color_tags={ 'blue': 'seen', 'orange': 'unseen' },
                                xlabel=f'Number of {level}', title=f'Distribution of {level} (train and query)',
                                filename=os.path.join(plot_folder, f'train_query_{level}_colorbar.pdf'), hide_legend=True)
        plot_count_in_splits_as_colors( [ NamedCounts('Seen', [ counts['seen_keys'][level] ], ['blue']),
                                NamedCounts('Val unseen', [ counts['val_unseen_keys'][level] ], ['orange']),
                                NamedCounts('Test unseen', [ counts['test_unseen_keys'][level] ], ['orange'])], 
                                ranges, expand=True,
                                color_tags={ 'blue': 'seen', 'orange': 'unseen' },
                                xlabel=f'Number of {level}', title=f'Distribution of {level} (keys)',
                                filename=os.path.join(plot_folder, f'key_{level}_colorbar.pdf'))

        plot_count_in_splits_as_histogram( [ NamedCounts('Train', [ counts['train'][level]], ['blue']),
                                NamedCounts('Val seen&unseen', [ counts['val_seen_query'][level], counts['val_unseen_query'][level] ], ['blue', 'orange']),
                                NamedCounts('Test seen&unseen', [ counts['test_seen_query'][level], counts['test_unseen_query'][level] ], ['blue', 'orange']) ],
                                ranges, expand=True,
                                color_tags={ 'blue': 'seen', 'orange': 'unseen' },
                                ylabel=f'Number of {level}',
                                xlabel=f'Number of records', 
                                title=f'Distribution of {level} (train and query)',
                                filename=os.path.join(plot_folder, f'train_query_{level}_dist.pdf'))
        plot_count_in_splits_as_histogram( [ NamedCounts('Seen', [ counts['seen_keys'][level] ], ['blue']),
                                NamedCounts('Val unseen', [ counts['val_unseen_keys'][level] ], ['orange']),
                                NamedCounts('Test unseen', [ counts['test_unseen_keys'][level] ], ['orange'])], 
                                ranges, expand=True,
                                color_tags={ 'blue': 'seen', 'orange': 'unseen' },
                                ylabel=f'Number of {level}',
                                xlabel=f'Number of records', 
                                title=f'Distribution of {level} (keys)',
                                filename=os.path.join(plot_folder, f'key_{level}_dist.pdf'))

    counts = get_dist(trainval_level_counts)
    for level in levels:
        plot_count_in_splits_as_colors( [ NamedCounts('Train', [ counts['train'][level]], ['green']),
                                NamedCounts('Val', [ counts['val'][level] ], ['green']),
                                NamedCounts('Test', [ counts['test'][level] ], ['green']) ], 
                                ranges_colorbar, expand=True,
                                xlabel=f'Number of {level}', 
                                title=f'Distribution of {level}',
                                filename=os.path.join(plot_folder, f'trainvaltest_total_{level}_colorbar.pdf'))

        plot_count_in_splits_as_histogram( [ NamedCounts('Train', [ counts['train'][level]], ['green']),
                                NamedCounts('Val', [ counts['val'][level] ], ['green']),
                                NamedCounts('Test', [ counts['test'][level] ], ['green']) ], 
                                ranges, expand=True,
                                ylabel=f'Number of {level}',
                                xlabel=f'Number of records', 
                                title=f'Distribution of {level}',
                                filename=os.path.join(plot_folder, f'trainvaltest_total_{level}_dist.pdf'))

if __name__ == "__main__":
    main()
