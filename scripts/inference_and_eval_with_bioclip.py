import json
import json
import os
import contextlib
import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from inference_and_eval_with_bioclip_with_image_to_image import compute_accuracy
from bioscanclip.util.dataset import load_dataloader, load_bioscan_dataloader_with_train_seen_and_separate_keys, \
    load_bioscan_dataloader_all_small_splits
import open_clip
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai_templates = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]


def make_txt_features(model, classnames, templates):
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    with torch.no_grad():
        txt_features = []
        for classname in tqdm(classnames):
            classname = " ".join(word for word in classname.split("_") if word)
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(DEVICE)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            txt_features.append(class_embedding)
        txt_features = torch.stack(txt_features, dim=1).to(DEVICE)
    return txt_features


def make_prediction(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1]
    return pred.cpu().tolist(), target.cpu().tolist()


def get_all_unique_species_from_dataloader(dataloader):
    all_species = []
    species_to_other = {}
    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch['species']

        for idx, species in enumerate(label_batch['species']):
            if species not in species_to_other.keys():
                species_to_other[species] = {"order": label_batch['order'][idx], "family": label_batch['family'][idx],
                                             "genus": label_batch['genus'][idx], 'species': species}

    all_species = list(set(all_species))
    all_species.sort()
    return all_species, species_to_other


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.suppress


def encode_image_feature_and_calculate_accuracy(model, txt_features, query_dataloader, all_species):
    # for image feature
    autocast = get_autocast("amp")
    pbar = tqdm(query_dataloader)
    all_pred = []
    all_gt = []
    for batch in pbar:
        pbar.set_description("Encode image feature...")
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

        targets = torch.tensor([all_species.index(species) for species in label_batch['species']]).to(DEVICE)

        image_input_batch = image_input_batch.to(DEVICE)
        with autocast():
            image_features = model.encode_image(image_input_batch)
            image_features = F.normalize(image_features, dim=-1)
            logits = model.logit_scale.exp() * image_features @ txt_features

        # Measure accuracy
        pred, target = make_prediction(logits, targets, topk=(1, 3, 5))
        all_pred = all_pred + pred
        all_gt = all_gt + target

    compute_accuracy(all_pred, all_gt)


@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join(
        args.visualization.output_dir, args.model_config.model_output_name, "features_and_prediction"
    )
    os.makedirs(folder_for_saving, exist_ok=True)

    # initialize model
    print("Initialize model...")
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.to(DEVICE)

    # Load data
    args.model_config.batch_size = 24
    _, _, _, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
        args)
    _, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

    all_species, species_to_other = get_all_unique_species_from_dataloader(all_keys_dataloader)
    classnames = [name.replace("_", " ") for name in all_species]
    txt_features_of_all_species = make_txt_features(model, classnames, openai_templates)

    print("For seen test: ")
    encode_image_feature_and_calculate_accuracy(model, txt_features_of_all_species, seen_test_dataloader, all_species)
    print()

    print("For unseen test: ")
    encode_image_feature_and_calculate_accuracy(model, txt_features_of_all_species, unseen_test_dataloader, all_species)


if __name__ == "__main__":
    main()
