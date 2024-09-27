import copy
import json
import os
import datetime
import hydra
import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm import tqdm

from inference_and_eval import get_features_and_label, inference_and_print_result
from model.simple_clip import load_clip_model
from util.dataset import load_bioscan_dataloader_all_small_splits


def main_process(rank: int, world_size: int, args):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")

    # # Set up for debug, delete when you see it!

    args.debug_flag = False
    # # Set up for debug, delete when you see it!

    # special set up for train on INSECT dataset
    args.model_config.batch_size = 200
    args.model_config.epochs = 80

    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    # current_datetime = datetime.datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")
    train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader = load_bioscan_dataloader_all_small_splits(args, world_size=None, rank=None)

    print("Initialize model...")
    model = load_clip_model(args)
    checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')
    model.load_state_dict(checkpoint)

    model = model.to(device)

    folder_path_to_save_features_and_other_infor = "bioscan_1M_features_and_meta_that_ready_for_bzsl"
    os.makedirs(folder_path_to_save_features_and_other_infor)


    # keys of the dict include ['file_name_list', 'encoded_dna_feature', 'encoded_image_feature', 'encoded_language_feature', 'averaged_feature', 'concatenated_feature', 'label_list', 'all_key_features', 'all_key_features_label']
    train_dict_for_feature = get_features_and_label(train_seen_dataloader, model, device, for_key_set=False)
    seen_val_dict_for_feature = get_features_and_label(seen_val_dataloader, model, device, for_key_set=False)
    unseen_val_dict_for_feature = get_features_and_label(unseen_val_dataloader, model, device, for_key_set=False)
    seen_test_dict_for_feature = get_features_and_label(seen_test_dataloader, model, device, for_key_set=False)
    unseen_test_dict_for_feature = get_features_and_label(unseen_test_dataloader, model, device, for_key_set=False)

    all_split_dict = {'train': train_dict_for_feature,
                      'val_seen': seen_val_dict_for_feature,
                      'val_unseen': unseen_val_dict_for_feature,
                      'test_seen': seen_test_dict_for_feature,
                      'test_unseen': unseen_test_dict_for_feature
                      }

    dict_of_loc_lists = {}
    all_image_feature = None
    all_dna_feature = None
    all_species_list = []
    current_len = 0
    for split_name in all_split_dict.keys():
        curr_dict = all_split_dict[split_name]
        if all_image_feature is None:
            all_image_feature = curr_dict['encoded_image_feature']
        else:
            all_image_feature = np.concatenate([all_image_feature, curr_dict['encoded_image_feature']],
                           axis=0)

        if all_dna_feature is None:
            all_dna_feature = curr_dict['encoded_dna_feature']
        else:
            all_dna_feature = np.concatenate([all_dna_feature, curr_dict['encoded_dna_feature']],
                           axis=0)

        for i in curr_dict['label_list']:
            all_species_list.append(i['species'])


        curr_loc = list(range(len(curr_dict['label_list'])))
        curr_loc = [x + current_len for x in curr_loc]
        dict_of_loc_lists[split_name + "_loc"] = curr_loc
        current_len = current_len + len(curr_loc)

    unique_species = list(set(all_species_list))
    unique_species.sort()

    all_label_list = [unique_species.index(species) for species in all_species_list]

    dict_to_json = dict_of_loc_lists

    # dict_to_json['att'] = all_dna_feature
    # dict_to_json['feature'] = all_image_feature
    dict_to_json['labels'] = all_label_list
    dict_to_json['species'] = all_species_list
    
    # save metadata
    folder_to_save_dict = os.path.join(args.project_root_path, 'bioscan_data_in_insect_format')
    os.makedirs(folder_to_save_dict, exist_ok=True)
    file_path = os.path.join(folder_to_save_dict, 'bioscan_1m_data_in_insect_format.json')
    with open(file_path, 'w') as json_file:
        json.dump(dict_to_json, json_file, indent=4)
        
    # save image feature feature
    image_embedd_path = os.path.join(folder_to_save_dict, 'image_embed.csv')
    dna_embedd_path = os.path.join(folder_to_save_dict, 'dna_embed.csv')
    np.savetxt(image_embedd_path, all_image_feature, delimiter=',')
    np.savetxt(dna_embedd_path, all_dna_feature, delimiter=',')


@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
