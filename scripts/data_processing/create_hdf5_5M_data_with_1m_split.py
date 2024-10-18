import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

KEYS_OF_DATA_WE_WANT_TO_KEEP = ['barcode', 'dna_bin', 'family', 'genus', 'image', 'image_file', 'image_mask',
                                'language_tokens_attention_mask', 'language_tokens_input_ids',
                                'language_tokens_token_type_ids', 'order', 'processid', 'sampleid', 'species']


def find_1m_sample_id_in_5m_sample_id_for_each_splits(sample_id_from_1m_pre_train_split, bioscan_5m_hdf5_file):
    key_to_index_we_want = {}
    all_split_from_bioscan_5m = list(bioscan_5m_hdf5_file.keys())
    pbar = tqdm(all_split_from_bioscan_5m)
    set_1m = set(sample_id_from_1m_pre_train_split)
    for key in pbar:
        pbar.set_description(f"Processing {key}")
        curr_split_of_5m = bioscan_5m_hdf5_file[key]
        curr_split_sample_id_from_5m = curr_split_of_5m['sampleid'][:]
        curr_split_sample_id_from_5m = np.array(curr_split_sample_id_from_5m, dtype='S')
        pre_train_split_5m_sample_id_list = np.char.decode(curr_split_sample_id_from_5m, 'utf-8')
        indices_in_1m = []
        for index, sample_id in enumerate(pre_train_split_5m_sample_id_list):
            if sample_id in set_1m:
                set_1m.remove(sample_id)
                indices_in_1m.append(index)
        key_to_index_we_want[key] = indices_in_1m

    print(f"Number of sample id from 1m pre train split that are not in 5m: {len(set_1m)}")

    return key_to_index_we_want


def process_key(key, bioscan_5m_hdf5_file, key_to_index_we_want, splits):
    data = None
    curr_data = None
    for split in tqdm(splits, desc=f"Loading splits for {key}", leave=False):
        curr_data = bioscan_5m_hdf5_file[split][key][key_to_index_we_want[split]]
        if data is None:
            data = curr_data
        else:
            data = np.concatenate((data, bioscan_5m_hdf5_file[split][key][key_to_index_we_want[split]]), axis=0)
    return key, data


def create_new_hdf5_file_with_1m_pre_train_split(new_path, key_to_index_we_want, bioscan_5m_hdf5_file, num_processes=4):
    if os.path.exists(new_path):
        os.remove(new_path)

    new_hdf5_file = h5py.File(new_path, "w")
    splits = list(key_to_index_we_want.keys())
    new_split = new_hdf5_file.create_group('no_split_and_seen_train')

    with Pool(processes=num_processes) as pool:
        keys_to_process = list(KEYS_OF_DATA_WE_WANT_TO_KEEP)

        results = []
        for key in tqdm(keys_to_process, desc="Submitting tasks"):
            result = pool.apply_async(process_key, args=(key, bioscan_5m_hdf5_file, key_to_index_we_want, splits))
            results.append(result)

        for result in tqdm(results, desc="Processing results"):
            key, data = result.get()
            new_split.create_dataset(key, data=data)

    new_hdf5_file.close()


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    path_to_5m_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    bioscan_5m_hdf5_file = h5py.File(path_to_5m_hdf5, "r")

    path_to_1m_hdf5 = args.bioscan_data.path_to_hdf5_data
    bioscan_1m_hdf5_file = h5py.File(path_to_1m_hdf5, "r")
    pre_train_split_1m = bioscan_1m_hdf5_file["no_split_and_seen_train"]

    sample_id_from_1m_pre_train_split = pre_train_split_1m['sampleid'][:]
    sample_id_from_1m_pre_train_split = np.array(sample_id_from_1m_pre_train_split, dtype='S')

    sample_id_from_1m_pre_train_split = np.char.decode(sample_id_from_1m_pre_train_split, 'utf-8')

    key_to_index_we_want = find_1m_sample_id_in_5m_sample_id_for_each_splits(sample_id_from_1m_pre_train_split,
                                                                             bioscan_5m_hdf5_file)

    new_hdf5_file_path = os.path.join(args.bioscan_5m_data.dir, "bioscan_5m_with_1m_pre_train_split.hdf5")

    create_new_hdf5_file_with_1m_pre_train_split(new_hdf5_file_path, key_to_index_we_want, bioscan_5m_hdf5_file)

    bioscan_5m_hdf5_file.close()
    bioscan_1m_hdf5_file.close()


if __name__ == '__main__':
    main()
