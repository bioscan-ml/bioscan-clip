import io
import os
import csv

import cv2
import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bioscanclip.model.simple_clip import load_clip_model

LEVELS = ["order", "family", "genus", "species"]

# Reference and modified fromhttps://github.com/jacobgil/vit-explain
def rollout(attentions, discard_ratio, head_fusion="max", layer_idx=None):
    result = torch.eye(attentions[0].size(-1))

    if layer_idx is None:
        all_attentions = attentions[1:6]
    else:
        # all_attentions = attentions[layer_idx:layer_idx + 1]
        # all_attentions = attentions[1:layer_idx]+attentions[layer_idx + 1:]
        all_attentions = attentions[1:layer_idx+1]

    with torch.no_grad():
        for attention in all_attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="min",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.hooks = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                hook = module.register_forward_hook(self.get_attention)
                self.hooks.append(hook)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu().detach())

    def __call__(self, input_tensor, layer_idx=None):
        self.attentions.clear()
        with torch.no_grad():
            output = self.model(input_tensor)
        result = rollout(self.attentions, self.discard_ratio, self.head_fusion, layer_idx)
        self.attentions.clear()

        return result

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def encode_all_image(image_list, model, transform, device):
    for image in image_list:
        image = transform(image).unsqueeze(0).to(device)
        image_output = F.normalize(model(image), p=2, dim=-1)


def get_some_images_from_hdf5(hdf5_group, id_list, n=None):

    image_list = []; image_id_list = []
    if n is None: n = len(hdf5_group["image"])
    for idx in range(n):

        image_enc_padded = hdf5_group["image"][idx].astype(np.uint8)
        enc_length = hdf5_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc)).resize((256, 256))

        id = hdf5_group["image_file"][idx]
        if id in id_list:
            image_list.append(curr_image)
            image_id_list.append(str(id, "utf-8").split(".")[0])
    
    return image_list, image_id_list


def get_image_encoder(model, device):
    image_encoder = model.image_encoder
    image_encoder.eval()
    image_encoder.to(device)
    return image_encoder


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_vit_explaination(image_list, attn_rollout, transform, device, layer_idx=None):
    image_with_mask_list = []
    for idx, image in tqdm(enumerate(image_list), total=len(image_list)):
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)
            mask = attn_rollout(image_tensor, layer_idx=layer_idx)
        
        np_img = np.array(image)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        image_with_mask = show_mask_on_image(np_img, mask)
        image_with_mask_list.append(image_with_mask)

    attn_rollout.remove_hooks()
    return image_with_mask_list


def get_and_save_vit_explaination(image_list, image_id_list, attn_rollout, transform, device, layer_idx=None,
                                  folder_name="representation_visualization/before_contrastive_learning"):
    os.makedirs(folder_name, exist_ok=True)
    image_with_mask_list = []
    for image, image_id in tqdm(zip(image_list, image_id_list), total=len(image_list)):
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)
            mask = attn_rollout(image_tensor, layer_idx=layer_idx)

        np_img = np.array(image)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        image_with_mask = show_mask_on_image(np_img, mask)
        cv2.imwrite(f"{folder_name}/{image_id}.png", image_with_mask)
        image_with_mask_list.append(image_with_mask)
    attn_rollout.remove_hooks()
    return image_with_mask_list


@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:

    seen_flag = False; seen_string = "seen" if seen_flag else "unseen"
    val_flag = True; val_string = "val" if val_flag else "test"
    # save_path = os.path.join(args.project_root_path, "representation_visualization_test/macro/seen")
    save_path = os.path.join(args.project_root_path, f"representation_visualization/{seen_string}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load some images from the hdf5 file
    if args.model_config.dataset == "bioscan_5m":
        path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    else:
        path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

    it_config = OmegaConf.load(args.model_config.it_config_path)
    id_config = OmegaConf.load(args.model_config.id_config_path)
    idt_config = args.model_config

    # Init transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    id_image_dict = {}
    for l_id, level in enumerate(LEVELS):
        id_image_dict[level] = {}

        # read csv
        level_column = 1 + l_id * 2
        with open(os.path.join(args.project_root_path, f"logs/IDT_val_top1_{seen_string}_i2i_results.csv"), mode='r') as after_csvfile,\
            open(os.path.join(args.project_root_path, f"logs/before_val_top1_{seen_string}_i2i_results.csv"), mode='r') as before_csvfile:
            after_reader = csv.reader(after_csvfile, delimiter=',')
            next(after_reader)

            before_reader = csv.reader(before_csvfile, delimiter=',')
            next(before_reader)

            before_success_after_failed_id_list = []; before_failed_after_failed_id_list = []
            before_success_after_success_id_list = []; before_failed_after_success_id_list = []
            for after_row, before_row in zip(after_reader, before_reader):
                id = str.encode(after_row[0]+".jpg")

                if after_row[level_column] != after_row[level_column+1]:
                    if before_row[level_column] == before_row[level_column+1]:
                        before_success_after_failed_id_list.append(id)
                    else:
                        before_failed_after_failed_id_list.append(id)
                else:
                    if before_row[level_column] == before_row[level_column+1]:
                        before_success_after_success_id_list.append(id)
                    else:
                        before_failed_after_success_id_list.append(id)

        # Open the hdf5 file
        with  h5py.File(path_to_hdf5, "r", libver="latest") as hdf5_file:
            # For now just use train_seen data
            hdf5_group = hdf5_file[f"{val_string}_{seen_string}"]
            # Load some images from the hdf5 file
            failed_failed_image_list, failed_failed_image_id_list =\
                get_some_images_from_hdf5(hdf5_group, before_failed_after_failed_id_list)
            failed_success_image_list, failed_success_image_id_list =\
                get_some_images_from_hdf5(hdf5_group, before_failed_after_success_id_list)

            id_image_dict[level]["failed_failed"] = {
                "id": failed_failed_image_id_list,
                "image": failed_failed_image_list}
            id_image_dict[level]["failed_success"] = {
                "id": failed_success_image_id_list,
                "image": failed_success_image_list}

            success_failed_image_list, success_failed_image_id_list =\
                get_some_images_from_hdf5(hdf5_group, before_success_after_failed_id_list)
            success_success_image_list, success_success_image_id_list =\
                get_some_images_from_hdf5(hdf5_group, before_success_after_success_id_list)

            id_image_dict[level]["success_failed"] = {
                "id": success_failed_image_id_list,
                "image": success_failed_image_list}
            id_image_dict[level]["success_success"] = {
                "id": success_success_image_id_list,
                "image": success_success_image_list}


    for level in LEVELS:
        for before_label in ["failed", "success"]:
            for after_label in ["failed", "success"]:

                print("Initialize model...")
                args.model_config = idt_config
                model = load_clip_model(args, device)
                model.eval()
                # Get the image encoder
                image_encoder = get_image_encoder(model, device)
                for block in image_encoder.lora_vit.blocks:
                    block.attn.fused_attn = False

                print(f"get before contrastive learning {before_label}_{after_label} image list at {level} level")
                attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
                # before_failed_image_list = get_vit_explaination(failed_image_list, attn_rollout, transform, device)
                before_masked_image_list = get_and_save_vit_explaination(id_image_dict[level][f"{before_label}_{after_label}"]["image"], 
                    id_image_dict[level][f"{before_label}_{after_label}"]["id"], attn_rollout, transform, device, folder_name=\
                        os.path.join(save_path, f"{level}/{before_label}_{after_label}/before_contrastive_learning"))


                checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
                model.load_state_dict(checkpoint)

                model.eval()
                # Get the image encoder
                image_encoder = get_image_encoder(model, device)
                for block in image_encoder.lora_vit.blocks:
                    block.attn.fused_attn = False

                print(f"get (I,D,T) after contrastive learning {before_label}_{after_label} image list at {level} level")
                attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
                # after_success_image_list = get_vit_explaination(success_image_list, attn_rollout, transform, device)
                after_masked_image_list_idt = get_and_save_vit_explaination(id_image_dict[level][f"{before_label}_{after_label}"]["image"], 
                    id_image_dict[level][f"{before_label}_{after_label}"]["id"], attn_rollout, transform, device, folder_name=\
                        os.path.join(save_path, f"{level}/{before_label}_{after_label}/image_dna_text"))


                args.model_config = it_config 
                model = load_clip_model(args, device)
                checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
                model.load_state_dict(checkpoint)

                model.eval()
                # Get the image encoder
                image_encoder = get_image_encoder(model, device)
                for block in image_encoder.lora_vit.blocks:
                    block.attn.fused_attn = False

                print(f"get (I,T) after contrastive learning {before_label}_{after_label} image list at {level} level")
                attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
                # after_success_image_list = get_vit_explaination(success_image_list, attn_rollout, transform, device)
                after_masked_image_list_it = get_and_save_vit_explaination(id_image_dict[level][f"{before_label}_{after_label}"]["image"], 
                    id_image_dict[level][f"{before_label}_{after_label}"]["id"], attn_rollout, transform, device, folder_name=\
                        os.path.join(save_path, f"{level}/{before_label}_{after_label}/image_text"))

                args.model_config = id_config 
                model = load_clip_model(args, device)
                checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
                model.load_state_dict(checkpoint)

                model.eval()
                # Get the image encoder
                image_encoder = get_image_encoder(model, device)
                for block in image_encoder.lora_vit.blocks:
                    block.attn.fused_attn = False

                print(f"get (I,D) after contrastive learning {before_label}_{after_label} image list at {level} level")
                attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
                # after_success_image_list = get_vit_explaination(success_image_list, attn_rollout, transform, device)
                after_masked_image_list_id = get_and_save_vit_explaination(id_image_dict[level][f"{before_label}_{after_label}"]["image"], 
                    id_image_dict[level][f"{before_label}_{after_label}"]["id"], attn_rollout, transform, device, folder_name=\
                        os.path.join(save_path, f"{level}/{before_label}_{after_label}/image_dna"))


                print(f"Merge {before_label}_{after_label} images at {level} level")
                os.makedirs(os.path.join(save_path, f"{level}/{before_label}_{after_label}/origin"), exist_ok=True)
                os.makedirs(os.path.join(save_path, f"{level}/{before_label}_{after_label}/merge"), exist_ok=True)
                os.makedirs(os.path.join(save_path, f"{level}/{before_label}_{after_label}/merge_alignments"), exist_ok=True)

                image_id_list = id_image_dict[level][f"{before_label}_{after_label}"]["id"]
                image_list = id_image_dict[level][f"{before_label}_{after_label}"]["image"]

                for image, image_id, before_image, after_image_it, after_image_id, after_image_idt in \
                    tqdm(zip(image_list, image_id_list, before_masked_image_list, \
                             after_masked_image_list_it, after_masked_image_list_id, after_masked_image_list_idt), total=len(image_list)):

                    image = np.array(image)[:, :, ::-1]
                    cv2.imwrite(os.path.join(save_path, f"{level}/{before_label}_{after_label}/origin", f"{image_id}.png"), image)

                    height = image.shape[0]
                    separator = np.ones((height, 10, 3), dtype=np.uint8) * 255

                    combined_image = np.hstack((image, separator, before_image, separator, after_image_idt))
                    cv2.imwrite(os.path.join(save_path, f"{level}/{before_label}_{after_label}/merge", f"{image_id}.png"), combined_image)

                    combined_image = np.hstack((image, separator, before_image, separator, after_image_it, separator, after_image_id, separator, after_image_idt))
                    cv2.imwrite(os.path.join(save_path, f"{level}/{before_label}_{after_label}/merge_alignments", f"{image_id}.png"), combined_image)



    # for layer_idx in range(2,12):
    #     print(f"Layer {layer_idx}")

    #     attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
    #     get_and_save_vit_explaination(image_list, attn_rollout, transform, device, layer_idx=layer_idx,
    #                                   folder_name=os.path.join(save_path, f"remove_after_layer/{layer_idx}"))


if __name__ == "__main__":
    main()