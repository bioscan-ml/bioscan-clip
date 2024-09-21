import io
import os

import cv2
import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.model.simple_clip import load_clip_model


# Reference and modified fromhttps://github.com/jacobgil/vit-explain
def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions[1:]:
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
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def encode_all_image(image_list, model, transform, device):
    for image in image_list:
        image = transform(image).unsqueeze(0).to(device)
        image_output = F.normalize(model(image), p=2, dim=-1)


def get_some_images_from_hdf5(hdf5_group, n=100):
    image_list = []
    for idx in range(n):
        image_enc_padded = hdf5_group["image"][idx].astype(np.uint8)
        enc_length = hdf5_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        image_list.append(curr_image)
    return image_list


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


def get_and_save_vit_explaination(image_list, grad_rollout, transform, device,
                                  folder_name="representation_visualization/before_contrastive_learning"):
    os.makedirs(folder_name, exist_ok=True)
    for idx, image in tqdm(enumerate(image_list), total=len(image_list)):
        image_tensor = transform(image).unsqueeze(0).to(device)
        mask = grad_rollout(image_tensor)
        mask_255 = (mask * 255).astype(np.uint8)
        mask_255_image = Image.fromarray(mask_255)
        mask_255_image.save(f"{folder_name}/vit_explaination_mask_{idx}.png")

        np_img = np.array(image)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        image_with_mask = show_mask_on_image(np_img, mask)
        cv2.imwrite(f"{folder_name}/vit_explaination_image_with_mask_{idx}.png", image_with_mask)


@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    for head_fusion in ["mean", "max", "min"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load some images from the hdf5 file
        if args.model_config.dataset == "bioscan_5m":
            path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
        else:
            path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

        # Init transform
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Open the hdf5 file
        hdf5_file = h5py.File(path_to_hdf5, "r", libver="latest")
        # For now just use train_seen data
        hdf5_group = hdf5_file["train_seen"]

        # Load some images from the hdf5 file
        image_list = get_some_images_from_hdf5(hdf5_group)
        # Save these images into a folder call "representation_visualization/original_images"
        os.makedirs("representation_visualization/original_images", exist_ok=True)
        for idx, image in enumerate(image_list):
            image.save(f"representation_visualization/original_images/image_{idx}.png")

        print("Initialize model...")
        model = load_clip_model(args, device)
        model.eval()
        # Get the image encoder
        image_encoder = get_image_encoder(model, device)
        for block in image_encoder.lora_vit.blocks:
            block.attn.fused_attn = False

        attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion=head_fusion)
        get_and_save_vit_explaination(image_list, attn_rollout, transform, device,
                                      folder_name=os.path.join(args.project_root_path,
                                                               f"representation_visualization/{head_fusion}/before_contrastive_learning"))

        if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
            pass
        else:
            checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
            model.load_state_dict(checkpoint)

        model.eval()
        # Get the image encoder
        image_encoder = get_image_encoder(model, device)
        for block in image_encoder.lora_vit.blocks:
            block.attn.fused_attn = False

        attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion=head_fusion)
        get_and_save_vit_explaination(image_list, attn_rollout, transform, device,
                                      folder_name=os.path.join(args.project_root_path,
                                                               f"representation_visualization/{head_fusion}/after_contrastive_learning"))


if __name__ == "__main__":
    main()
