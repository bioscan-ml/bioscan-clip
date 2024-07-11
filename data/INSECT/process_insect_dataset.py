import os.path

import h5py
import numpy as np
from PIL import Image
import io
import pandas as pd


def save_list_of_images_into_hdf5(class_name, file_name):

    path_to_image = []
    for i in range(len(class_name)):
        curr_path = os.path.join('INSECT_images', "images", class_name[i], file_name[i])
        path_to_image.append(curr_path)
    with h5py.File('INSECT_images.hdf5', 'w') as hf:
        images_group = hf.create_group('images')
        for idx, file_path in enumerate(path_to_image):
            path = file_path + ".jpg"
            if not os.path.exists(path):
                path = file_path + ".JPG"

            with open(path, 'rb') as f:
                byte_data = np.frombuffer(f.read(), dtype=np.uint8)
            dataset_name = file_name[idx]
            images_group.create_dataset(dataset_name, data=byte_data)

def correct_format(weird_array):
    list_of_str = []
    for i in weird_array:
        curr_str = str(i[0][0])
        list_of_str.append(curr_str)
    return np.array(list_of_str)

def save_to_csv():
    path = "res101.mat"
    data_mat = sio.loadmat(path)
    labels = data_mat["labels"].ravel() - 1

    data = {
        'bold_ids': correct_format(data_mat['bold_ids']),
        'ids': correct_format(data_mat['ids']),
        'labels': labels,
        'species': correct_format(data_mat['species']),
        'nucleotides': correct_format(data_mat['nucleotides']),
    }
    df = pd.DataFrame(data)

    path = "att_splits.mat"
    splits_mat = sio.loadmat(path)
    trainval_loc = splits_mat["trainval_loc"].ravel() - 1
    train_loc = splits_mat["train_loc"].ravel() - 1
    val_loc = splits_mat["val_loc"].ravel() - 1
    test_seen_loc = splits_mat["test_seen_loc"].ravel() - 1
    test_unseen_loc = splits_mat["test_unseen_loc"].ravel() - 1

    trainval_loc_list = []
    train_loc_list = []
    val_loc_list = []
    test_seen_loc_list = []
    test_unseen_loc_list = []

    for i in range(len(correct_format(data_mat['bold_ids']))):
        if i in trainval_loc:
            trainval_loc_list.append(1)
        else:
            trainval_loc_list.append(0)

        if i in train_loc:
            train_loc_list.append(1)
        else:
            train_loc_list.append(0)

        if i in val_loc:
            val_loc_list.append(1)
        else:
            val_loc_list.append(0)

        if i in test_seen_loc:
            test_seen_loc_list.append(1)
        else:
            test_seen_loc_list.append(0)

        if i in test_unseen_loc:
            test_unseen_loc_list.append(1)
        else:
            test_unseen_loc_list.append(0)

    df['trainval'] = trainval_loc_list
    df['train'] = train_loc_list
    df['val'] = val_loc_list
    df['test_seen'] = test_seen_loc_list
    df['test_unseen'] = test_unseen_loc_list

    df.to_csv("INSECT_metadata.csv", index=False)


if __name__ == '__main__':
    save_to_csv()

    df = pd.read_csv("INSECT_metadata.csv", index_col=None)
    save_list_of_images_into_hdf5(df['species'].to_list(), df['ids'].to_list())

    # Code for check_images
    # with h5py.File('INSECT_images.hdf5', 'r') as hf:
    #     selected_dataset = hf['images'][df['ids'].to_list()[0]]
    #
    #     image = Image.open(io.BytesIO(selected_dataset[:]))
    #
    #     image.show()
