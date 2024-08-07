
## Introduction about the embedding hdf5 files

Here is showing the datasets stored in one of the embedding hdf5 file.

```shell
h5ls -r extracted_features_of_all_keys.hdf5
/                        Group
/encoded_dna_feature     Dataset {21118, 768}
/encoded_image_feature   Dataset {21118, 768}
/encoded_language_feature Dataset {21118, 768}
/family_list             Dataset {21118}
/file_name               Dataset {21118}
/genus_list              Dataset {21118}
/order_list              Dataset {21118}
/species_list            Dataset {21118}
```

As all of the datasets are used to store either a list of string or a Numpy array, they are very easy to read. For example:
```shell
embedding_hdf5 = h5py.File(embedding_hdf5_path, "r", libver="latest")
encoded_dna_feature = embedding_hdf5['embedding_hdf5'][:]
order_list = [item.decode("utf-8") for item in embedding_hdf5['order']]
```
