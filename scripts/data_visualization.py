import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import math
import pandas as pd
from brokenaxes import brokenaxes
from matplotlib.colors import LinearSegmentedColormap
import hydra
from omegaconf import DictConfig

def get_heatmap_y_axis_data(fred_list, pre_define_ranges):
    y_data_for_heatmap = [0 for i in range(len(pre_define_ranges))]
    for count in fred_list:
        for idx, specific_range in enumerate(pre_define_ranges):
            if count in specific_range:
                y_data_for_heatmap[idx] = y_data_for_heatmap[idx] + 1
    return y_data_for_heatmap


# def plot_species_count_in_each_species_split(all_species_count_in_seen,
#                                              all_species_count_in_val_unseen,
#                                              all_species_count_in_test_unseen):
#     seen_freq = [count for key, count in Counter(all_species_count_in_seen).items()]
#     val_unseen_freq = [count for key, count in Counter(all_species_count_in_val_unseen).items()]
#     test_unseen_freq = [count for key, count in Counter(all_species_count_in_test_unseen).items()]
#
#     range_list = [{'min': 2, 'max': 10}, {'min': 11, 'max': 20}, {'min': 21, 'max': 40}, {'min': 41, 'max': 80},
#                   {'min': 81, 'max': 160}, {'min': 160, 'max': 'inf'}]
#
#     data = {
#     }
#     for range in range_list:
#         range_max = range['max']
#         if range_max == 'inf':
#             range_max = float('inf')
#         if f"{range['min']}-{range['max']}" not in data.keys():
#             data[f"{range['min']}-{range['max']}"] = []
#         for freq_list in [test_unseen_freq, val_unseen_freq, seen_freq]:
#             count = len([value for value in freq_list if range['min'] <= value < range_max])
#             data[f"{range['min']}-{range['max']}"].append(count)
#
#     index = ['Test Unseen', 'Val unseen', 'Seen']
#
#     df = pd.DataFrame(data, index=index)
#
#     g = df.plot(kind='barh', stacked=True, color=sns.color_palette("pastel", n_colors=6))
#     plt.tick_params(axis='x', labelsize=28)
#     plt.tick_params(axis='y', labelsize=28)
#     plt.title('Distribution of species', fontsize=32)
#     plt.xlabel('Number of species', fontsize=32)
#     plt.ylabel('Split of species', fontsize=32)
#     plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., fontsize=32)
#
#     plt.show()
def plot_species_count_in_each_species_split(all_species_count_in_seen,
                                             all_species_count_in_val_unseen,
                                             all_species_count_in_test_unseen):
    # Generate frequency lists
    seen_freq = [count for key, count in Counter(all_species_count_in_seen).items()]
    val_unseen_freq = [count for key, count in Counter(all_species_count_in_val_unseen).items()]
    test_unseen_freq = [count for key, count in Counter(all_species_count_in_test_unseen).items()]

    # Define range list
    range_list = [{'min': 2, 'max': 10}, {'min': 11, 'max': 20}, {'min': 21, 'max': 40}, {'min': 41, 'max': 80},
                  {'min': 81, 'max': 160}, {'min': 161, 'max': 2714}]

    # Initialize data structure
    data = {}
    for range_def in range_list:
        range_max = range_def['max']
        range_label = f"{range_def['min']}-{range_def['max'] if range_def['max'] != float('inf') else 'inf'}"
        data[range_label] = []
        for freq_list in [seen_freq, val_unseen_freq, test_unseen_freq]:
            count = len([value for value in freq_list if range_def['min'] <= value <= range_max])
            data[range_label].append(count)

    # Data preparation
    index = ['Seen', 'Val Unseen', 'Test Unseen']
    colors = sns.color_palette("pastel", n_colors=len(range_list))
    start_color = colors[0]  # Light blue
    end_color = colors[1]  # Orange
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color], N=len(range_list))

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.85)
    bottom = np.array([0, 0, 0])
    for i, (range_label, counts) in enumerate(data.items()):
        color = cmap(i / (len(range_list) - 1))
        ax.barh(index, counts, left=bottom, color=color, label=range_label)
        bottom += np.array(counts)

    ax.set_xlabel('Number of species', fontsize=32)
    ax.set_title('Distribution of species', fontsize=32)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=24)

    plt.tight_layout()
    plt.show()

# def plot_species_count_in_each_species_split(all_species_count_in_seen,
#                                              all_species_count_in_val_unseen,
#                                              all_species_count_in_test_unseen):
#     # Generate frequency lists
#     seen_freq = [count for key, count in Counter(all_species_count_in_seen).items()]
#     val_unseen_freq = [count for key, count in Counter(all_species_count_in_val_unseen).items()]
#     test_unseen_freq = [count for key, count in Counter(all_species_count_in_test_unseen).items()]
#
#     # Define range list
#     range_list = [{'min': 2, 'max': 10}, {'min': 11, 'max': 20}, {'min': 21, 'max': 40}, {'min': 41, 'max': 80},
#                   {'min': 81, 'max': 160}, {'min': 161, 'max': 2714}]
#
#     # Initialize data structure
#     data = {}
#     for range_def in range_list:
#         range_label = f"{range_def['min']}-{range_def['max'] if range_def['max'] != float('inf') else 'inf'}"
#         data[range_label] = []
#         for freq_list in [seen_freq, val_unseen_freq, test_unseen_freq]:
#             count = len([value for value in freq_list if range_def['min'] <= value <= range_def['max']])
#             data[range_label].append(count)
#
#     # Data preparation
#     index = ['Seen', 'Val Unseen', 'Test Unseen']
#     colors = sns.color_palette("pastel", n_colors=len(range_list))
#     start_color = colors[0]  # Light blue
#     end_color = colors[1]  # Orange
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color], N=len(range_list))
#
#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 6))
#     width = 0.35  # width of the bar
#     indices = np.arange(len(index))  # the x locations for the groups
#
#     # Bottom array to keep track of the bottom of the stacked bars
#     bottoms = np.zeros(len(index))
#
#     # Plot each range
#     count = 0
#     for range_label, counts in data.items():
#         color = cmap(count / (len(range_list) - 1))
#         count += 1
#         ax.bar(indices, counts, width, bottom=bottoms, label=range_label, color=color)
#         bottoms += np.array(counts)
#
#     ax.set_ylabel('Number of species')
#     ax.set_title('Distribution of species by category')
#     ax.set_xticks(indices)
#     ax.set_xticklabels(index)
#     ax.legend(title='Species count ranges')
#
#     plt.show()


def plot_histogram(list_of_data, list_of_label, threshold):
    categories = []
    all_data = []
    for idx, freq_of_split in enumerate(list_of_data):
        categories = categories + [list_of_label[idx]] * len(freq_of_split)
        all_data = all_data + freq_of_split

    df = pd.DataFrame({'Value': all_data, 'Category': categories})
    colors = sns.color_palette("pastel", n_colors=3)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    sns.boxplot(data=df, x='Category', y='Value', ax=ax1, palette=colors)
    sns.boxplot(data=df, x='Category', y='Value', ax=ax2, palette=colors)
    ax1.set_ylim([50, 700])
    ax2.set_ylim([0, 20])
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax1.xaxis.tick_bottom()
    plt.xticks(fontsize=18)

    plt.xlabel('Split of species', fontsize=18)
    # plt.ylabel('Frequency of count of each species over the species split.', fontsize=16)
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.143, wspace=0.4, hspace=0.4)
    plt.gcf().set_size_inches(10, 8)

    plt.show()


def plot_heatmap_to_show_how_distribution_of_data_in_species_level(all_species_count_in_seen,
                                                                   all_species_count_in_val_unseen,
                                                                   all_species_count_in_test_unseen):
    # Plot a heat map to show the distirbution of the species
    seen_freq = [count for _, count in Counter(all_species_count_in_seen).items()]
    val_unseen_freq = [count for _, count in Counter(all_species_count_in_val_unseen).items()]
    test_unseen_freq = [count for _, count in Counter(all_species_count_in_test_unseen).items()]

    list_of_label = ['Seen', 'Val Unseen', 'Test Unseen']
    list_of_data = [seen_freq, val_unseen_freq, test_unseen_freq]
    plot_histogram(list_of_data, list_of_label, threshold=30)


@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    # Get all species list in a dict
    h5file = h5py.File(args.bioscan_data.path_to_hdf5_data, 'r')
    all_split_key = list(h5file.keys())
    print(all_split_key)

    species_dict = {}
    splits_to_skip = ['no_split_and_seen_train', 'no_split', 'all_keys', 'single_species']
    all_species_list = []
    all_seen_species = []
    all_val_unseen_species = []
    all_test_unseen_species = []

    for split_name in all_split_key:
        if split_name in splits_to_skip:
            continue
        curr_species_list = [item.decode('utf-8') for item in h5file[split_name]['species']]
        species_dict[split_name] = {'list_of_species': curr_species_list,
                                    'number_of_species': len(set(curr_species_list)),
                                    'number_of_records': len(curr_species_list)}
        print(
            f'Curr split: {split_name} ||number of species: {len(set(curr_species_list))}||number of records: {len(curr_species_list)}')
        all_species_list = all_species_list + curr_species_list

    all_unique_species = list(set(all_species_list))
    all_species_count_dict = {}

    all_seen_species = species_dict['seen_keys']['list_of_species'] + species_dict['train_seen'][
        'list_of_species'] + species_dict['val_seen']['list_of_species'] + species_dict['test_seen']['list_of_species']

    all_val_unseen_species = species_dict['val_unseen']['list_of_species'] + species_dict['val_unseen_keys'][
        'list_of_species']
    all_test_unseen_species = species_dict['test_unseen']['list_of_species'] + \
                              species_dict['test_unseen_keys']['list_of_species']

    for species in all_species_list:
        if species not in all_species_count_dict.keys():
            all_species_count_dict[species] = 0
        else:
            all_species_count_dict[species] = all_species_count_dict[species] + 1

    dict_for_count = {'maj': {'seen': 0, 'val_unseen': 0, 'test_unseen': 0, 'total': 0},
                      'min': {'seen': 0, 'val_unseen': 0, 'test_unseen': 0, 'total': 0}}

    for species in all_species_count_dict.keys():
        if all_species_count_dict[species] >= 9:
            dict_for_count['maj']['total'] = dict_for_count['maj']['total'] + 1
            if species in all_seen_species:
                dict_for_count['maj']['seen'] = dict_for_count['maj']['seen'] + 1
            if species in all_val_unseen_species:
                dict_for_count['maj']['val_unseen'] = dict_for_count['maj']['val_unseen'] + 1
            if species in all_test_unseen_species:
                dict_for_count['maj']['test_unseen'] = dict_for_count['maj']['test_unseen'] + 1
        if all_species_count_dict[species] < 9:
            dict_for_count['min']['total'] = dict_for_count['min']['total'] + 1
            if species in all_seen_species:
                dict_for_count['min']['seen'] = dict_for_count['min']['seen'] + 1
            if species in all_val_unseen_species:
                dict_for_count['min']['val_unseen'] = dict_for_count['min']['val_unseen'] + 1
            if species in all_test_unseen_species:
                dict_for_count['min']['test_unseen'] = dict_for_count['min']['test_unseen'] + 1

    # print(dict_for_count['min']['seen'])
    # exit()
    print()
    print('For maj species (with at least 9 samples in the species)')
    print(f"{dict_for_count['maj']['seen'] * 1.0 / dict_for_count['maj']['total'] * 100}% species in seen")
    print(f"{dict_for_count['maj']['val_unseen'] * 1.0 / dict_for_count['maj']['total'] * 100}% species in val_unseen")
    print(
        f"{dict_for_count['maj']['test_unseen'] * 1.0 / dict_for_count['maj']['total'] * 100}% species in test_unseen")
    print()
    print('For min species (with less than 9 samples in the species)')
    print(f"{dict_for_count['min']['seen'] * 1.0 / dict_for_count['min']['total'] * 100}% species in seen")
    print(f"{dict_for_count['min']['val_unseen'] * 1.0 / dict_for_count['min']['total'] * 100}% species in val_unseen")
    print(
        f"{dict_for_count['min']['test_unseen'] * 1.0 / dict_for_count['min']['total'] * 100}% species in test_unseen")

    print()
    print('For seen species')
    # all_seen_species = species_dict['seen_keys']['list_of_species'] + species_dict['train_seen']['list_of_species'] + species_dict['val_seen']['list_of_species'] + species_dict['test_seen']['list_of_species']
    print(
        f"{len(species_dict['train_seen']['list_of_species']) * 1.0 / len(all_seen_species) * 100}% species in train seen")
    print(
        f"{len(species_dict['val_seen']['list_of_species']) * 1.0 / len(all_seen_species) * 100}% species in val_seen seen")
    print(
        f"{len(species_dict['test_seen']['list_of_species']) * 1.0 / len(all_seen_species) * 100}% species in test_seen seen")
    print(
        f"{len(species_dict['seen_keys']['list_of_species']) * 1.0 / len(all_seen_species) * 100}% species in test_seen seen")

    print()
    print('For val_unseen species')
    aall_val_unseen_species = species_dict['val_unseen']['list_of_species'] + species_dict['val_unseen_keys'][
        'list_of_species']

    print(
        f"{len(species_dict['val_unseen']['list_of_species']) * 1.0 / len(all_val_unseen_species) * 100}% species in val_unseen(query)")
    print(
        f"{len(species_dict['val_unseen_keys']['list_of_species']) * 1.0 / len(all_val_unseen_species) * 100}% species in val_unseen_keys")

    print()
    print('For test_unseen species')
    all_test_unseen_species = species_dict['test_unseen']['list_of_species'] + \
                              species_dict['test_unseen_keys']['list_of_species']
    print(
        f"{len(species_dict['test_unseen']['list_of_species']) * 1.0 / len(all_test_unseen_species) * 100}% species in test_unseen(query)")
    print(
        f"{len(species_dict['test_unseen_keys']['list_of_species']) * 1.0 / len(all_test_unseen_species) * 100}% species in test_unseen_keys")

    all_species_count_in_seen = species_dict['seen_keys']['list_of_species'] + species_dict['train_seen'][
        'list_of_species'] + species_dict['val_seen']['list_of_species'] + species_dict['test_seen']['list_of_species']

    all_species_count_in_val_unseen = species_dict['val_unseen']['list_of_species'] + species_dict['val_unseen_keys'][
        'list_of_species']
    all_species_count_in_test_unseen = species_dict['test_unseen']['list_of_species'] + \
                                       species_dict['test_unseen_keys'][
                                           'list_of_species']

    all_species_list = all_species_count_in_seen + all_species_count_in_val_unseen + all_species_count_in_test_unseen

    # print(Counter(all_species_list))
    # exit()

    plot_species_count_in_each_species_split(all_species_count_in_seen,
                                             all_species_count_in_val_unseen,
                                             all_species_count_in_test_unseen)
    # plot_heatmap_to_show_how_distribution_of_data_in_species_level(all_species_count_in_seen,
    #                                                                all_species_count_in_val_unseen,
    #                                                                all_species_count_in_test_unseen)

if __name__ == "__main__":
    main()



