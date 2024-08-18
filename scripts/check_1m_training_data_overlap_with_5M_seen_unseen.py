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

    print(hdf5_5m.keys())

    hdf5_1m_pre_train = hdf5_1m["no_split_and_seen_train"]



    hdf5_5m_seen = hdf5_5m["seen_keys"]
    hdf5_5m_unseen = hdf5_5m["unseen_keys"]


    classes = ['order', 'family', 'genus', 'species']

    for key in classes:
        hdf5_1m_pre_train_classes = [item.decode("utf-8") for item in hdf5_1m_pre_train[key][:]]
        classes_seen_from_5m = [item.decode("utf-8") for item in hdf5_5m_seen[key][:]]
        classes_unseen_from_5m = [item.decode("utf-8") for item in hdf5_5m_unseen[key][:]]

        # get unique classes
        classes_1m = set(hdf5_1m_pre_train_classes)
        seen_classes_5m = set(classes_seen_from_5m)
        unseen_classes_5m = set(classes_unseen_from_5m)

        # covert all classes to lowercase, and replace space with underscore
        classes_1m = {item.lower().replace(" ", "_") for item in classes_1m}
        seen_classes_5m = {item.lower().replace(" ", "_") for item in seen_classes_5m}
        unseen_classes_5m = {item.lower().replace(" ", "_") for item in unseen_classes_5m}

        # get 1M classed that are in 5M seen and unseen
        classes_1m_in_5m_seen = classes_1m.intersection(seen_classes_5m)
        classes_1m_in_5m_unseen = classes_1m.intersection(unseen_classes_5m)

        print(f'Number of unique {key} in 1M: ', len(classes_1m))
        print(f'Number of unique {key} in 1M train split and 5M seen: ', len(seen_classes_5m))
        print(f'Number of unique {key} in 1M train split and 5M unseen: ', len(unseen_classes_5m))

        #         write to json
        with open(os.path.join(args.project_root_path, f'logs/1m_unique_{key}_in_train_split_exist_in_5M_seen.json'), 'w') as f:
            json.dump(list(classes_1m_in_5m_seen), f)
        with open(os.path.join(args.project_root_path, f'logs/1m_unique_{key}_in_train_split_exist_in_5M_unseen.json'), 'w') as f:
            json.dump(list(classes_1m_in_5m_unseen), f)

if __name__ == "__main__":
    main()
