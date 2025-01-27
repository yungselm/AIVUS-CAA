import math
import random
from glob import glob
from typing import Tuple, List, Any, Dict
from joblib import Parallel, delayed

import albumentations as A
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from joblib import Parallel
from sklearn.model_selection import train_test_split


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 img_files: List[str],
                 mask_files: List[str],
                 batch_size: int,
                 img_size: (int, int),
                 img_channels: int = 1,
                 mask_channels: int = 1,
                 augmentation_p: float = 0.5,
                 random_rotate_p: float = 0.3,
                 vertical_flip_p: float = 0.5,
                 horizontal_flip_p: float = 0.5,
                 shuffle=True,
                 hue_p=0.5,
                 contrast_p=0.5,
                 brightness_p=0.5,
                 hue_shift_limit: int = 20,
                 sat_shift_limit: int = 30,
                 val_shift_limit: int = 20,
                 contrast_limit: float = 0.2,
                 brightness_limit: float = 0.2,
                 ):
        # nfile = glob(data_path + '/NAC/*.nii.gz')
        # train_files, val_files = train_test_split(nfile, test_size=val_size, random_state=seed)
        img_data, mask_data = reg_data_prep(img_files, mask_files)
        self.img_paths = np.array(img_data)
        self.mask_paths = np.array(mask_data)
        print(img_data.shape, mask_data.shape)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation_p = augmentation_p
        self.transform = A.Compose([
            A.Rotate(limit=180, p=random_rotate_p),
            A.VerticalFlip(p=vertical_flip_p),
            A.HorizontalFlip(p=horizontal_flip_p),
            # A.CenterCrop(p=p_center_crop, height=img_size[0], width=img_size[1]),
            A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                 sat_shift_limit=sat_shift_limit,
                                 val_shift_limit=val_shift_limit,
                                 p=hue_p),
            # A.RandomContrast(limit=contrast_limit, p=contrast_p),
            # A.RandomBrightness(limit=brightness_limit, p=brightness_p),
        ], p=self.augmentation_p)
        self.img_size = img_size
        self.img_channel = img_channels
        self.mask_channel = mask_channels
        # self.cutmix_p = cutmix_p
        # self.p_mosaic = p_mosaic
        # self.beta = beta
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.img_paths)).astype(np.int32)
            self.img_paths, self.mask_paths = self.img_paths[indices], self.mask_paths[indices]

    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_img = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size, *self.img_size, self.img_channel), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.img_size, self.mask_channel), dtype=np.uint8)

        for i, (img, mask) in enumerate(zip(batch_img, batch_mask)):
            # img = cv2.resize(img, self.img_size)
            # mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            augmented = self.transform(image=img.astype(np.float32), mask=mask.astype(np.uint8))
            img, mask = augmented['image'], augmented['mask']
            x[i] = img
            y[i] = mask

        # y = y.reshape((self.batch_size, *self.img_size, 1)) / 255  # normalization is done for all the samples
        y = np.concatenate([y] * self.mask_channel, axis=-1)
        x = x / 255
        return x, y


################################################################################
def load_nii_file(fpath):
    arr = nib.load(fpath)
    arr = np.asanyarray(arr.dataobj)
    if len(arr.shape) == 2:
        arr = arr[..., None]
    arr = np.swapaxes(arr, 0, 2)
    return arr



################################################################################
def reg_data_prep(img_list: List[str], mask_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data = Parallel(n_jobs=1)(
        delayed(read_data)(img_path, mask_path) for img_path, mask_path in zip(img_list, mask_list))
    img_data_set, mask_data_set = list(zip(*data))
    mask_data_set = np.concatenate(mask_data_set, axis=0)
    img_data_set = np.concatenate(img_data_set, axis=0)
    return img_data_set, mask_data_set


def read_data(img_path, mask_path):
    img_data = load_nii_file(img_path)
    mask_data = load_nii_file(mask_path)
    mask_data = np.expand_dims(mask_data, axis=3)
    img_data = np.expand_dims(img_data, axis=3)
    return img_data, mask_data
