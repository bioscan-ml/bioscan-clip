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
from util.dataset import load_bioscan_dataloader, load_bioscan_dataloader_with_train_seen_and_separate_keys
import open_clip
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image_key_features(model, all_keys_dataloader):
    key_labels = {}
    with torch.no_grad():
        key_features = []
        autocast = get_autocast("amp")
        pbar = tqdm(all_keys_dataloader)
        for batch in pbar:
            pbar.set_description("Encode key feature...")
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            image_input_batch = image_input_batch.to(DEVICE)
            with autocast():
                image_features = model.encode_image(image_input_batch)
                image_features = F.normalize(image_features, dim=-1)
            for image_feature in image_features:
                # print(image_feature.shape)
                # exit()
                key_features.append(image_feature)
            for key in label_batch.keys():
                if key not in key_labels:
                    key_labels[key] = []
                key_labels[key] = key_labels[key] + label_batch[key]

        key_features = torch.stack(key_features, dim=1).to(DEVICE)
    return key_features, key_labels


def make_prediction(output, key_species, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    pred = pred.T.cpu().numpy()
    all_prediction = []
    for top_5_for_curr_sample in pred:
        predicted_top_5_species = [key_species[index] for index in top_5_for_curr_sample]
        all_prediction.append(predicted_top_5_species)

    return all_prediction

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.suppress

def compute_accuracy(predictions, ground_truth):
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    total_samples = len(ground_truth)
    class_correct_and_total_count = {}
    for i, truth in enumerate(ground_truth):
        # Top-1 accuracy check
        if truth not in class_correct_and_total_count.keys():
            class_correct_and_total_count[truth] = {'top1_c':0.0, 'top3_c':0.0, 'top5_c':0.0, 'total':0.0}
        class_correct_and_total_count[truth]['total'] = class_correct_and_total_count[truth]['total'] + 1

        if predictions[i][0] == truth:
            top_1_correct += 1
            class_correct_and_total_count[truth]['top1_c'] = class_correct_and_total_count[truth]['top1_c'] + 1

        if truth in predictions[i][0:3]:
            top_3_correct += 1
            class_correct_and_total_count[truth]['top3_c'] = class_correct_and_total_count[truth]['top3_c'] + 1

        # Top-5 accuracy check
        if truth in predictions[i]:
            top_5_correct += 1
            class_correct_and_total_count[truth]['top5_c'] = class_correct_and_total_count[truth]['top5_c'] + 1

    # Calculate accuracies
    top_1_accuracy = top_1_correct / total_samples
    top_3_accuracy = top_3_correct / total_samples
    top_5_accuracy = top_5_correct / total_samples
    print('For micro acc')
    print(f"Top-1 Accuracy: {top_1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {top_3_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")

    top_1_class_acc_list = []
    top_3_class_acc_list = []
    top_5_class_acc_list = []



    for i in class_correct_and_total_count.keys():
        top_1_class_acc_list.append(class_correct_and_total_count[i]['top1_c']*1.0/class_correct_and_total_count[i]['total'])
        top_3_class_acc_list.append(
            class_correct_and_total_count[i]['top3_c'] * 1.0 /class_correct_and_total_count[i]['total'])
        top_5_class_acc_list.append(
            class_correct_and_total_count[i]['top5_c'] * 1.0 /class_correct_and_total_count[i]['total'])

    macro_top_1_accuracy = sum(top_1_class_acc_list) * 1.0 / len(top_1_class_acc_list)
    macro_top_3_accuracy = sum(top_3_class_acc_list) * 1.0 / len(top_1_class_acc_list)
    macro_top_5_accuracy = sum(top_5_class_acc_list) * 1.0 / len(top_1_class_acc_list)

    print('For macro acc')
    print(f"Top-1 Accuracy: {macro_top_1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {macro_top_3_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {macro_top_5_accuracy * 100:.2f}%")

def encode_image_feature_and_calculate_accuracy(model, key_features, key_labels, query_dataloader):
    # for image feature
    autocast = get_autocast("amp")
    pbar = tqdm(query_dataloader)
    key_species = key_labels['species']

    all_predictions = []
    all_gt_species = []
    for batch in pbar:
        pbar.set_description("Encode image feature...")
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

        gt_species = label_batch['species']
        all_gt_species = all_gt_species + gt_species
        image_input_batch = image_input_batch.to(DEVICE)
        with autocast():
            image_features = model.encode_image(image_input_batch)
            image_features = F.normalize(image_features, dim=-1)
            logits = model.logit_scale.exp() * image_features @ key_features

        predictions = make_prediction(logits, key_species, topk=(1, 5))
        all_predictions = all_predictions + predictions

    compute_accuracy(all_predictions, all_gt_species)



@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
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
    print("Initialize dataloader...")
    args.model_config.batch_size = 24
    _, _, _, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader = load_bioscan_dataloader_with_train_seen_and_separate_keys(
        args, for_pretrain=False)
    _, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_bioscan_dataloader(args)

    key_features, key_labels = make_image_key_features(model, all_keys_dataloader)

    print("For seen val: ")
    encode_image_feature_and_calculate_accuracy(model, key_features, key_labels, seen_val_dataloader)
    print()

    print("For unseen val: ")
    encode_image_feature_and_calculate_accuracy(model, key_features, key_labels, unseen_val_dataloader)


if __name__ == "__main__":
    main()
