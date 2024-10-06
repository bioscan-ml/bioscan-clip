import os
import h5py
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from multiprocessing import Pool, current_process


def process_large_split(args):
    split, file_path, start_idx, end_idx = args
    file = h5py.File(file_path, "r")
    data_list = []
    file_name_list = [item.decode("utf-8") for item in file[split]["image_file"][start_idx:end_idx]]

    pbar = tqdm(file_name_list, position=0, desc=f"Process {current_process()._identity[0]}: {split}")
    for file_name in pbar:
        idx = file_name_list.index(file_name) + start_idx
        data = {
            'image_file': file_name,
            'sampleid': file[split]["sampleid"][idx].decode("utf-8"),
            'barcode': file[split]["barcode"][idx].decode("utf-8"),
            'dna_bin': file[split]["dna_bin"][idx].decode("utf-8"),
            'order': file[split]["order"][idx].decode("utf-8"),
            'family': file[split]["family"][idx].decode("utf-8"),
            'genus': file[split]["genus"][idx].decode("utf-8"),
            'species': file[split]["species"][idx].decode("utf-8"),
            'split': split
        }
        data_list.append(data)

    return data_list


def process_other_splits(args):
    split, file_path = args
    file = h5py.File(file_path, "r")
    data_list = []
    file_name_list = [item.decode("utf-8") for item in file[split]["image_file"]]

    for file_name in file_name_list:
        idx = file_name_list.index(file_name)
        data = {
            'image_file': file_name,
            'sampleid': file[split]["sampleid"][idx].decode("utf-8"),
            'barcode': file[split]["barcode"][idx].decode("utf-8"),
            'dna_bin': file[split]["dna_bin"][idx].decode("utf-8"),
            'order': file[split]["order"][idx].decode("utf-8"),
            'family': file[split]["family"][idx].decode("utf-8"),
            'genus': file[split]["genus"][idx].decode("utf-8"),
            'species': file[split]["species"][idx].decode("utf-8"),
            'split': split
        }
        data_list.append(data)

    return data_list


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

    splits = ['seen_keys', 'single_species', 'test_seen', 'test_unseen', 'test_unseen_keys', 'train_seen', 'val_seen',
              'val_unseen', 'val_unseen_keys']
    large_split = 'no_split'
    file = h5py.File(path_to_hdf5, "r")
    num_items = len(file[large_split]["image_file"])
    segment_size = num_items // 4  # Assuming 4 segments to parallel process large split

    with Pool(processes=15) as pool:  # One more than segments for other splits
        large_split_args = [(large_split, path_to_hdf5, i * segment_size, (i + 1) * segment_size) for i in range(4)]
        large_results = pool.map(process_large_split, large_split_args)
        other_results = pool.map(process_other_splits, [(split, path_to_hdf5) for split in splits])

    all_data = [item for sublist in large_results + other_results for item in sublist]
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(args.bioscan_data.dir, 'data_with_split.csv'), index=False)


if __name__ == '__main__':
    main()
