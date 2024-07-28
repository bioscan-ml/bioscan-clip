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

TODO: add instructions on how to read the data into Numpy.
