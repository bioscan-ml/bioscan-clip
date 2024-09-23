import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import math
import pandas as pd
from brokenaxes import brokenaxes
import json
import copy
from scipy.interpolate import make_interp_spline

def load_hdf5_data(data_hdf5_path):
    h5file = h5py.File(data_hdf5_path, 'r')
    return h5file

def avg_list(l):

    return sum(l) * 1.0 / len(l)

def get_avg_acc_by_key_count(count_list, acc_list):
    record_num_to_acc = {}
    for record_number_of_species_in_key_set, acc in zip(count_list, acc_list):
        if record_number_of_species_in_key_set not in record_num_to_acc.keys():
            record_num_to_acc[record_number_of_species_in_key_set] = []
        record_num_to_acc[record_number_of_species_in_key_set].append(acc)

    number_of_record_list = []
    averaged_acc = []

    for record_number_of_species_in_key_set in record_num_to_acc.keys():
        number_of_record_list.append(record_number_of_species_in_key_set)
        averaged_acc.append(avg_list(record_num_to_acc[record_number_of_species_in_key_set]))
    return number_of_record_list, averaged_acc




def plot_scatterplot(species_2_query_count_and_acc, image_or_dna_as_key):
    seen_species_count_list = []
    seen_species_acc_list = []
    unseen_species_count_list = []
    unseen_species_acc_list = []



    for species in species_2_query_count_and_acc.keys():
        if 'seen' in species_2_query_count_and_acc[species].keys():
            seen_species_count_list.append(species_2_query_count_and_acc[species]['key_count'])
            seen_species_acc_list.append(species_2_query_count_and_acc[species]['seen'])
        if 'unseen' in species_2_query_count_and_acc[species].keys():
            unseen_species_count_list.append(species_2_query_count_and_acc[species]['key_count'])
            unseen_species_acc_list.append(species_2_query_count_and_acc[species]['unseen'])

    seen_species_count_list, seen_species_acc_list = get_avg_acc_by_key_count(seen_species_count_list, seen_species_acc_list)
    unseen_species_count_list, unseen_species_acc_list = get_avg_acc_by_key_count(unseen_species_count_list,
                                                                                unseen_species_acc_list)

    # Plotting both seen and unseen species data
    dot_size = 300
    fonr_size = 18
    colors = sns.color_palette("pastel", n_colors=2)
    plt.figure(figsize=(16, 10))
    plt.scatter(seen_species_count_list, seen_species_acc_list, color=colors[0], label='Seen Species', s=dot_size)
    plt.scatter(unseen_species_count_list, unseen_species_acc_list, color=colors[1], label='Unseen Species', s=dot_size)

    # plt.title(f'Record Count of Species in Key set vs. Probability for Seen and Unseen Species, using {image_or_dna_as_key} Feature as Key', fontsize=fonr_size, pad=30)
    plt.xlabel('Number of records of the species in the key set', fontsize=fonr_size+6)
    plt.ylabel('Probability', fontsize=fonr_size+6)
    plt.tick_params(axis='x', labelsize=fonr_size-2)
    plt.tick_params(axis='y', labelsize=fonr_size-2)
    plt.legend(fontsize=fonr_size+6)
    # plt.xlim(left=2, right=300)
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.show()


def plot_multiple_scatterplot(per_class_acc_dict, all_keys_species, query_feature_list, key_feature_list, seen_and_unseen, k_list, levels):
    # get dict for species to count
    species_2_key_record_count = {}
    for species in all_keys_species:
        if species not in species_2_key_record_count.keys():
            species_2_key_record_count[species] = {}
            species_2_key_record_count[species]['key_count'] = 0
        species_2_key_record_count[species]['key_count'] = species_2_key_record_count[species]['key_count'] + 1

    # For the combination of query and key
    for query_type in query_feature_list:
        for key_type in key_feature_list:
            print(f'Query: {query_type}, Key: {key_type}')
            species_2_query_count_and_acc = copy.deepcopy(species_2_key_record_count)
            for seen_or_unseen in seen_and_unseen:
                for k in k_list:
                    for level in levels:
                        curr_acc_dict = per_class_acc_dict[query_type][key_type][seen_or_unseen][k][level]
                        for species in curr_acc_dict.keys():
                            species_2_query_count_and_acc[species][seen_or_unseen] = curr_acc_dict[species]

            if key_type == 'encoded_image_feature':
                image_or_dna_as_key = 'Image'
            else:
                image_or_dna_as_key = 'DNA'
            plot_scatterplot(species_2_query_count_and_acc, image_or_dna_as_key)

def load_per_class_acc(per_class_acc_path):
    with open(per_class_acc_path) as json_file:
        per_class_acc = json.load(json_file)
    return per_class_acc


if __name__ == '__main__':
    per_class_acc_path = 'extracted_embedding/bioscan_1m/image_dna_text_4gpu/per_class_acc_val.json'
    per_class_acc_dict = load_per_class_acc(per_class_acc_path)

    query_feature_list = ['encoded_image_feature', 'encoded_dna_feature']
    key_feature_list = ['encoded_image_feature', 'encoded_dna_feature']
    seen_and_unseen = ['seen', 'unseen']
    k_list = ['1']
    levels = ['species']

    data_hdf5_path = 'data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5'
    # Get all species list in a dict
    data_h5file = load_hdf5_data(data_hdf5_path)

    all_keys_species = [item.decode('utf-8') for item in data_h5file['all_keys']['species']]

    plot_multiple_scatterplot(per_class_acc_dict, all_keys_species, query_feature_list, key_feature_list,
                              seen_and_unseen, k_list, levels)





