import json
import os

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from bioscanclip.util.dataset import load_dataloader


def load_all_seen_species_name_and_create_label_map(train_seen_dataloader):
    all_seen_species = []
    species_to_other_labels = {}

    for batch in train_seen_dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_seen_species = all_seen_species + label_batch['species']
        for curr_idx in range(len(label_batch['species'])):
            if label_batch['species'][curr_idx] not in species_to_other_labels.keys():
                species_to_other_labels[label_batch['species'][curr_idx]] = {'order': label_batch['order'][curr_idx],
                                                                             'family': label_batch['family'][curr_idx],
                                                                             'genus': label_batch['genus'][curr_idx]}
    return species_to_other_labels

def calculate_accuracies(predictions, ground_truths):
    correct_predictions = sum(p == gt for p, gt in zip(predictions, ground_truths))
    micro_accuracy = correct_predictions / len(predictions)

    class_correct_counts = {}
    class_actual_counts = {}
    for gt in set(ground_truths):
        class_correct_counts[gt] = sum((p == gt) and (gt == g) for p, g in zip(predictions, ground_truths))
        class_actual_counts[gt] = sum(gt == g for g in ground_truths)

    class_accuracies = [class_correct_counts[gt] / class_actual_counts[gt] if class_actual_counts[gt] > 0 else 0 for gt
                        in class_actual_counts]
    macro_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0


    print(f"Micro acc: {micro_accuracy}")
    print(f"Macro acc: {macro_accuracy}")

    return micro_accuracy, macro_accuracy


def get_other_labels_list(species_list, specie_to_other):
    order_list = []
    family_list = []
    genus_list = []
    for sp in species_list:
        order_list.append(specie_to_other[sp]['order'])
        family_list.append(specie_to_other[sp]['family'])
        genus_list.append(specie_to_other[sp]['genus'])

    return [order_list, family_list, genus_list, species_list]

def main_process(rank: int, world_size: int, args):
    pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(
        args, world_size=world_size, rank=rank)
    species_to_other_dict = load_all_seen_species_name_and_create_label_map(all_keys_dataloader)
    filename = os.path.join(args.project_root_path ,'BIOSCAN_1M_pred_and_gt.json')

    with open(filename, 'r') as f:
        pred_and_gt = json.load(f)

    for key in pred_and_gt.keys():
        print(f"{key}: {len(pred_and_gt[key])}")

    seen_pred_in_multi_order = get_other_labels_list(pred_and_gt['seen_pred'], species_to_other_dict)
    seen_gt_in_multi_order = get_other_labels_list(pred_and_gt['seen_gt'], species_to_other_dict)
    unseen_pred_in_multi_order = get_other_labels_list(pred_and_gt['unseen_pred'], species_to_other_dict)
    unseen_gt_in_multi_order = get_other_labels_list(pred_and_gt['unseen_gt'], species_to_other_dict)

    print("For seen")
    print("For order")
    index = 0
    calculate_accuracies(seen_pred_in_multi_order[index], seen_gt_in_multi_order[index])
    print("For family")
    index = 1
    calculate_accuracies(seen_pred_in_multi_order[index], seen_gt_in_multi_order[index])
    print("For genus")
    index = 2
    calculate_accuracies(seen_pred_in_multi_order[index], seen_gt_in_multi_order[index])
    print("For species")
    index = 3
    calculate_accuracies(seen_pred_in_multi_order[index], seen_gt_in_multi_order[index])


    print()
    print("For unseen")
    # calculate_accuracies(pred_and_gt['unseen_pred'], pred_and_gt['unseen_gt'])
    print("For order")
    index = 0
    calculate_accuracies(unseen_pred_in_multi_order[index], unseen_gt_in_multi_order[index])
    print("For family")
    index = 1
    calculate_accuracies(unseen_pred_in_multi_order[index], unseen_gt_in_multi_order[index])
    print("For genus")
    index = 2
    calculate_accuracies(unseen_pred_in_multi_order[index], unseen_gt_in_multi_order[index])
    print("For species")
    index = 3
    calculate_accuracies(unseen_pred_in_multi_order[index], unseen_gt_in_multi_order[index])

    pass



@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)




if __name__ == '__main__':
    main()
