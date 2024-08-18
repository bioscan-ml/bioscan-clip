import h5py
import io
import os
import random
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

def get_image(dataset_image, dataset_image_mask, processid_to_index, query_id):
    idx = processid_to_index[query_id]
    image_enc_padded = dataset_image[idx]
    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    print(args.project_root_path)
    dataset_hdf5_all_key = h5py.File(args.bioscan_5m_data.path_to_hdf5_data, "r", libver="latest")['all_keys']

    # you may put the embedding hdf5 somewhere else. you can modify the path if you want
    embedding_hdf5_path_all_key = os.path.join(args.project_root_path, "extracted_embedding/bioscan_5m/trained_with_bioscan_5m_image_dna_text/extracted_features_of_all_keys.hdf5")
    embedding_hdf5_all_key = h5py.File(embedding_hdf5_path_all_key, "r", libver="latest")


    # Load processid, image, image_mask from hdf5 file.
    dataset_processid_list = [item.decode("utf-8") for item in dataset_hdf5_all_key["processid"][:]]
    dataset_image = dataset_hdf5_all_key["image"][:].astype(np.uint8)
    dataset_image_mask = dataset_hdf5_all_key["image_mask"][:]
    processid_to_index = {pid: idx for idx, pid in enumerate(dataset_processid_list)}

    # Random select a processid from the embedding hdf5 file.
    processid_from_embedding_hdf5 = [item.decode("utf-8") for item in embedding_hdf5_all_key["processid"][:]]
    random_query_id = random.choice(processid_from_embedding_hdf5)

    image = get_image(dataset_image, dataset_image_mask, processid_to_index, random_query_id)

    image.show()

if __name__ == "__main__":

    main()
