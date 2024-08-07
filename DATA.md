## Data Structure in HDF5 Format

The data is stored in HDF5 format with the following structure. Each dataset contains multiple groups, each representing different splits of the data.

### Group Structure
Each group represents a specific data split and contains several datasets. The groups are organized as follows:

- `all_keys`: Contains all data that will be used as key during the evaluation.
- `val_seen`: Contains seen query data for validation.
- `test_seen`: Contains seen query data for testing.
- `seen_keys`: Contains seen data that will be used as key during the evaluation. Note, for BIOSCAN-5M, these data are also used for training.
- `test_unseen`: Contains unseen test data.
- `val_unseen`: Contains unseen validation data.
- `unseen_keys`: Contains unseen data that will be used as key during the evaluation.
- `no_split_and_seen_train`: All data that will be used for contrastive pretrain.

Notably, there are some slight differences in the group structure of the BIOSCAN-1M and BIOSCAN-5M data, but they are fundamentally consistent.

### Dataset Structure

Each group contains several datasets:

- `image`: Stores the image data as byte arrays.
- `image_mask`: Stores the length of each image byte array.
- `barcode`: Stores DNA barcode sequences.
- `family`: Stores the family classification of each sample.
- `genus`: Stores the genus classification of each sample.
- `order`: Stores the order classification of each sample.
- `sampleid`: Stores the sample IDs.
- `species`: Stores the species classification of each sample.
- `processid`: Stores the process IDs for each sample.
- `language_tokens_attention_mask`: Stores the attention masks for language tokens.
- `language_tokens_input_ids`: Stores the input IDs for language tokens.
- `language_tokens_token_type_ids`: Stores the token type IDs for language tokens.
- `image_file`: Stores the filenames of the images.


### Content of each split

Here is a view of the BIOSCAN-1M's `all_key` split's content.
```shell
h5ls -r BioScan_data_in_splits.hdf5

/                        Group
/all_keys                Group
/all_keys/barcode        Dataset {21118}
/all_keys/dna_features   Dataset {21118, 768}
/all_keys/family         Dataset {21118}
/all_keys/genus          Dataset {21118}
/all_keys/image          Dataset {21118, 24027}
/all_keys/image_features Dataset {21118, 512}
/all_keys/image_file     Dataset {21118}
/all_keys/image_mask     Dataset {21118}
/all_keys/language_tokens_attention_mask Dataset {21118, 20}
/all_keys/language_tokens_input_ids Dataset {21118, 20}
/all_keys/language_tokens_token_type_ids Dataset {21118, 20}
/all_keys/order          Dataset {21118}
/all_keys/sampleid       Dataset {21118}
/all_keys/species        Dataset {21118}
...
```

Most of the datasets store lists of encoded strings. To read them, you can use:
```shell
hdf5_split_group = h5py.File(hdf5_inputs_path, "r", libver="latest")['all_keys']
list_of_barcode = [item.decode("utf-8") for item in hdf5_split_group["barcode"][:]]
```

The `image_features` and `dna_features` stored in the dataset hdf5 files are pre-extracted by models without contrastive learning. We used them to get borderline results. You shouldn't need them, but to read them, you can:
```shell
image_features = hdf5_split_group["image_features"][:].astype(np.float32)
dna_features = hdf5_split_group["dna_features"][:].astype(np.float32)
```

The `language_tokens_attention_mask`, `language_tokens_input_ids` and `language_tokens_token_type_ids` are tokens of strings concatenated by the `order`, `family`, `genus` and `species` of each sample; we used them as language input when we perform contrastive training with BERT-small. We read them by:
```shell
language_input_ids = hdf5_split_group["language_tokens_input_ids"][:]
language_token_type_ids = hdf5_split_group["language_tokens_token_type_ids"][:]
language_attention_mask = hdf5_split_group["language_tokens_attention_mask"][:]
```

We stored images by converting the images into byte-encoded data with padding and record the lengths of each encoded image. This allows us to store the images in an HDF5 file. The encoded images are stored in the image dataset, while the lengths of the images are recorded in image_mask. To read them, you can reference:

```shell
image_enc_padded = hdf5_split_group["image"][idx].astype(np.uint8)
enc_length = hdf5_split_group["image_mask"][idx]
image_enc = image_enc_padded[:enc_length]
curr_image = Image.open(io.BytesIO(image_enc))
```

