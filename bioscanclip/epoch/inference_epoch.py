from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from bioscanclip.epoch.eval_epoch import convert_label_dict_to_list_of_dict
import torch


def get_feature_and_label(dataloader, model, device, for_open_clip=False, multi_gpu=False):
    encoded_image_feature_list = []
    encoded_dna_feature_list = []
    encoded_text_feature_list = []
    label_list = []
    file_name_list =[]
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Encoding features")
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            if for_open_clip:
                language_input = input_ids
            else:
                language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                                  'attention_mask': attention_mask.to(device)}

            image_output, dna_output, language_output, logit_scale, logit_bias = model(image_input_batch.to(device),
                                                                                       dna_input_batch.to(device),
                                                                                       language_input)

            if image_output is not None:
                encoded_image_feature_list = encoded_image_feature_list + F.normalize(image_output, dim=-1).cpu().tolist()
            if dna_output is not None:
                encoded_dna_feature_list = encoded_dna_feature_list + F.normalize(dna_output, dim=-1).cpu().tolist()
            if language_output is not None:
                encoded_text_feature_list = encoded_text_feature_list + F.normalize(language_output, dim=-1).cpu().tolist()




            label_list = label_list + convert_label_dict_to_list_of_dict(label_batch)
            file_name_list = file_name_list + list(processid_batch)

    if len(encoded_image_feature_list) == 0:
        encoded_image_feature_list = None
    else:
        encoded_image_feature_list = np.array(encoded_image_feature_list)
    if len(encoded_dna_feature_list) == 0:
        encoded_dna_feature_list = None
    else:
        encoded_dna_feature_list = np.array(encoded_dna_feature_list)
    if len(encoded_text_feature_list) == 0:
        encoded_text_feature_list = None
    else:
        encoded_text_feature_list = np.array(encoded_text_feature_list)

    return file_name_list, encoded_image_feature_list, encoded_dna_feature_list, encoded_text_feature_list, label_list