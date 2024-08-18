import h5py
import io
import json
import os
import csv
import random
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
import faiss
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import torch
from omegaconf import DictConfig
from sklearn.metrics import silhouette_samples
from umap import UMAP
from PIL import Image

from bioscanclip.epoch.inference_epoch import get_feature_and_label
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import Table, categorical_cmap

@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    print(args.project_root_path)

    path_to_1M_hdf5 = args.bioscan_data.path_to_hdf5_data
    path_to_5M_hdf5 = args.bioscan_5m_data.path_to_hdf5_data

    hdf5_1m = h5py.File(path_to_1M_hdf5, "r")
    hdf5_5m = h5py.File(path_to_5M_hdf5, "r")

    hdf5_1m_pre_train = hdf5_1m["no_split_and_seen_train"]
    hdf5_5m_pre_train = hdf5_5m["no_split_and_seen_train"]

    classes = ['order', 'family', 'genus', 'species']

    for key in classes:
        hdf5_1m_pre_train_classes = [item.decode("utf-8") for item in hdf5_1m_pre_train[key][:]]
        hdf5_5m_pre_train_classes = [item.decode("utf-8") for item in hdf5_5m_pre_train[key][:]]

        # get unique classes
        classes_1m = set(hdf5_1m_pre_train_classes)
        classes_5m = set(hdf5_5m_pre_train_classes)

        # covert all classes to lowercase, and replace space with underscore
        classes_1m = {item.lower().replace(" ", "_") for item in classes_1m}
        classes_5m = {item.lower().replace(" ", "_") for item in classes_5m}

        # get classes that are in 1M but not in 5M
        classes_1m_not_in_5m = classes_1m - classes_5m
        classes_5m_not_in_1m = classes_5m - classes_1m

        print(f'Number of unique {key} in 1M: ', len(classes_1m))
        print(f'Number of unique {key} in 5M: ', len(classes_5m))


        print(f"Number of {key} in 1M but not in 5M: ", len(classes_1m_not_in_5m))
        print(f"Number of {key} in 5M but not in 1M: ", len(classes_5m_not_in_1m))
        print()
        # Save this unique classes not exist in both datasets' training data to a json file in the unique_classes folder
        path = os.path.join(args.project_root_path, "unique_classes")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{key}_unique_classes_in_train_split_exist_in_1M_not_in_5M.json"), "w") as f:
            json.dump(list(classes_1m_not_in_5m), f)
        with open(os.path.join(path, f"{key}_unique_classes_in_train_split_exist_in_5M_not_in_1M.json"), "w") as f:
            json.dump(list(classes_5m_not_in_1m), f)

if __name__ == "__main__":
    main()
