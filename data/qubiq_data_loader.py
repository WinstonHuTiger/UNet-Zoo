import os
import numpy as np
import pandas as pd
import logging
import h5py
import random
import nibabel as nib
import cv2 as cv

# import utils

THRESHOLD_BACKGROUND = {
    'brain-tumor': 2e-5,
    'brain-growth': 0.0,
    'kidney': -1025.,
    'prostate': 0.0
}

NANNOTATOR = {
    'brain-tumor': 3,
    'brain-growth': 7,
    # TODO
    'kidney': 3,
    'prostate': 6,
}

NCLASS = {
    'brain-tumor': 3,
    'brain-growth': 1,
    # TODO
    'kidney': 1,
    'prostate': 2,

}
MEANS = {
    'brain-tumor': 501.699,
    # TODO
    'brain-growth': 569.1935,
    'kidney': -383.4882,
    'prostate': 429.64
}

STDS = {
    'brain-tumor': 252.760,
    # TODO
    'brain-growth': 189.9989,
    'kidney': 472.0944,
    'prostate': 300.0039
}

target_image_size = {
    "brain-tumor": 256,
    "brain-growth": 256,
    "kidney": 512,
    "prostate": 640
}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        x, y = sample
        assert type(x) is np.ndarray and type(y) is np.ndarray
        if random.random() < 0.5:
            x = x[:, :, -1::-1]
            y = y[:, :, -1::-1]

        return (x, y)


class RandomVerticalFlip(object):

    def __call__(self, sample):
        x, y = sample
        assert type(x) is np.ndarray and type(y) is np.ndarray
        if random.random() < 0.5:
            x = x[:, -1::-1, :]
            y = y[:, -1::-1, :]

        return (x, y)


class RandomResizedCrop:
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        x = sample['image']
        y = sample['lable']

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))

        return {
            'image': x,
            'label': y
        }


def prepare_data(root, dataset, task, output, output_file):
    NCLASS = NANNOTATOR[dataset]
    train_dir = os.path.join(root, 'train', 'training_data_v2', dataset, 'Training')
    train_label_dir = train_dir

    val_dir = os.path.join(root, 'val', 'validation_data_v2', dataset, "Validation")
    val_label_dir = val_dir

    test_dir = os.path.join(root, 'test', 'test_QUBIQ', dataset, 'Testing')
    test_label_dir = test_dir

    data_dirs = [train_dir, val_dir]
    label_dirs = [train_label_dir, val_label_dir]
    data_paths = {}
    label_paths = {}
    # print(label_dirs)

    for mode, data_dir in enumerate(data_dirs):
        # print(data_dir)
        img_paths = []
        labels = []
        with os.scandir(data_dir) as dirs:
            for subdir in dirs:
                # case
                if subdir.is_dir() is not True:
                    continue
                # discard cases with empty label
                if mode == 0 and dataset == 'brain-tumor' and task == 2 and subdir.name[-2:] in ['01', '03', '06', '07',
                                                                                                 '11']:
                    continue
                with os.scandir(subdir) as ssdir:
                    tasks = []
                    for ii in range(NCLASS):
                        tasks.append([])

                    for ff in ssdir:
                        if ff.name == 'image.nii.gz':
                            img_paths.append(ff.path)
                        else:
                            for ii in range(NCLASS):
                                # task(ii+1)
                                if ff.name[0:6] == 'task' + '{:0>2d}'.format(ii + 1):
                                    # if mode == 2:
                                    #    print(ff.path)
                                    tasks[ii].append(ff.path)
                    # sort to align the annotators
                    labels.append([sorted(x) for x in tasks])

        data_paths[mode] = img_paths
        label_paths[mode] = labels

    hdf5_file = h5py.File(output_file, "w")
    images = {}
    labels = {}
    groups = {}
    mean = MEANS[dataset]
    std = STDS[dataset]

    for i in ["train", 'val']:
        images[i] = []
        labels[i] = []
        groups[i] = hdf5_file.create_group(i)

    for mode, tt in enumerate(["train", 'val']):
        temp_images = []
        for image in data_paths[mode]:
            x = nib.load(image).get_fdata().astype(float)
            thres = THRESHOLD_BACKGROUND[dataset]
            fore = x > thres
            x[fore] -= mean
            x[fore] /= std

            if tt == "test":
                if dataset == "prostate":
                    x = np.squeeze(x)
                elif dataset == "brain-tumor":
                    x = np.transpose(x, [2, 0, 1])
                else:
                    x = np.array([x]).astype(float)
            temp_images.append(x)

        temp_labels = []

        for lb_task in label_paths[mode]:
            # print("lb_task", lb_task[task])
            y = []
            # print(lb_task)
            for lb in lb_task[task]:
                # print(lb_task[task])
                y.append(nib.load(lb).get_fdata().astype(float))
            # print("y", y)
            if len(y) == 0:
                print(lb_task[task])
            if output == "threshold":
                y = np.mean(np.stack(y), axis=0, keepdims=False)
                y_thres = []
                for ii in range(NANNOTATOR[dataset]):
                    y_thres.append(y >= (float(ii + 1) / NANNOTATOR[dataset]))
                y = y_thres
            elif output == 'annotator':
                y = np.stack(y)
            else:
                y = np.mean(np.stack(y), axis=0, keepdims=False)
                # for cross entropy
                y = np.floor(y * 10).astype(np.long)
            temp_labels.append(y)
        labels[tt] = []

        for i, image in enumerate(temp_images):
            x, y = __transform(tt, dataset, image, temp_labels[i])
            # x = cv.resize(x, dsize=(128, 128))
            # y = cv.resize(y, dsize=(128, 128))
            images[tt].append(x)
            labels[tt].append(y)

    for tt in ['train', "val"]:
        groups[tt].create_dataset("images", data=np.asarray(images[tt], dtype=np.float))
        groups[tt].create_dataset("labels", data=np.asarray(labels[tt], dtype=np.float))
    hdf5_file.close()


def __transform(mode, dataset, x, y):
    # transpose from HWC to CHW
    if dataset == 'brain-tumor':
        x = np.transpose(x, [2, 0, 1])
        y = np.array(y)

        target_size = target_image_size[dataset]
        c1, c2, c3 = x.shape
        needed_len = (target_size -c3)//2
        x = np.pad(x, ((0, 0), (needed_len, needed_len), (needed_len, needed_len)), "minimum")
        y = np.pad(y, ((0, 0), (needed_len, needed_len), (needed_len, needed_len)), "minimum")


    elif dataset == 'brain-growth':
        x = np.array([x])
        y = np.array(y)

    elif dataset == 'kidney':

        x = np.array([x])

        y = np.array(y)
        target_size = target_image_size[dataset]
        c1, c2, c3 = x.shape
        needed_len = (target_size -c3)//2
        x = np.pad(x, ((0, 0), (needed_len, needed_len +1 ), (needed_len, needed_len+1)), "minimum")
        y = np.pad(y, ((0, 0), (needed_len, needed_len +1), (needed_len, needed_len +1)), "minimum")

    elif dataset == 'prostate':
        x = np.array([x])
        y = np.squeeze(y)
        if mode != 'test':
            if x.shape[1] == 960:
                x = x[:, 160:800, :]
                y = y[:, 160:800, :]
        x = x[:, :, :, 0]

    x = np.transpose(x, [1, 2, 0])
    y = np.transpose(y, [1, 2, 0])
    return x, y


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def load_and_process_data(root,
                          dataset, task, output,
                          preprocessing_folder,
                          force_overwrite=True):
    if dataset not in ['brain-tumor', 'brain-growth', 'kidney', 'prostate']:
        raise NotImplementedError

    data_file_name = "data_quibq.hdf5"
    data_file_path = os.path.join(preprocessing_folder, data_file_name)
    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(root, dataset, task, output, data_file_path)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == "__main__":
    input_file = r"D:\dev_x\cv_uncertainty\qubiq"
    preproc_folder = "D:\dev_x\phiseg_log"
    dataset = "kidney"
    task = 0
    output = "annotator"
    d = load_and_process_data(input_file, dataset, task, output, preproc_folder)
