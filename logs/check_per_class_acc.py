import json
from IPython.display import display
from IPython.display import JSON
if __name__ == '__main__':
    # read json file to data
    with open('per_class_acc.json') as f:
        data = json.load(f)

    species_to_acc = {}

    for seen_or_unseen in ['seen', 'unseen']:
        group_of_acc = data['encoded_image_feature']['encoded_dna_feature'][seen_or_unseen]['1']['species']
        for key in group_of_acc.keys():
            if key not in species_to_acc:
                lower_key = key.lower().replace(" ", "_")
                species_to_acc[lower_key] = group_of_acc[key]

    # Read species_list_of_1m_train.json
    with open('species_list_of_1m_train.json') as f:
        species_list_of_1m_train = json.load(f)

    # Calculate the average accuracy of 1M training data for species in 1M training data
    acc_list = []
    for species in species_list_of_1m_train:
        lower_species = species.lower().replace(" ", "_")
        if lower_species in species_to_acc:
            acc_list.append(species_to_acc[lower_species])
    avg_acc = sum(acc_list) / len(acc_list)
    print(f"Average accuracy for species in 1M training data: {avg_acc}")

    # Calculate the average accuracy of 1M training data for species not in 1M training data
    acc_list = []
    for species in species_to_acc.keys():
        if species not in species_list_of_1m_train:
            acc_list.append(species_to_acc[species])
    avg_acc = sum(acc_list) / len(acc_list)
    print(f"Average accuracy for species not in 1M training data: {avg_acc}")


