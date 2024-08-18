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
    print()

    path_to_1M_hdf5 = args.bioscan_data.path_to_hdf5_data
    path_to_5m_hdf5 = args.bioscan_5m_data.path_to_hdf5_data

    hdf5_1m = h5py.File(path_to_1M_hdf5, "r")

    # get_species_list_of_1m 's no_split_and_seen_train split
    hdf5_1m_pre_train = hdf5_1m["no_split_and_seen_train"]
    species_list_of_1m_train = [item.decode("utf-8").lower().replace(" ", "_") for item in hdf5_1m_pre_train["species"][:]]

    with open(os.path.join(args.project_root_path, 'species_list_of_1m_train.json'), 'w') as f:
        json.dump(species_list_of_1m_train, f)


if __name__ == "__main__":
    main()
