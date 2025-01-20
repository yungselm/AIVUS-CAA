import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from deep_utils import DirUtils
from deep_utils import split_extension
from tensorflow.keras.models import load_model

from configs import *
from metrics import dice_score_tf


def load_nii_file(fpath):
    arr = nib.load(fpath)
    arr = np.asanyarray(arr.dataobj)
    if len(arr.shape) == 2:
        arr = arr[..., None]
    arr = np.swapaxes(arr, 0, 2)
    return arr


exp_number = 1
suffix = BEST_SUFFIX #LAST_SUFFIX
parser = ArgumentParser()

parser.add_argument("--input_path", default="output/images", help="path to the images")
parser.add_argument("--output_path", default="output/masks", help="where predicted masks will be saved")

args = parser.parse_args()

pred_batch_size = 256
SAVE_PATH = f"models/exp_{exp_number}/{TRIAL_IDENTIFIER}.h5"

if __name__ == '__main__':
    model = load_model(split_extension(SAVE_PATH, suffix=suffix),  custom_objects={"dice_score_tf": dice_score_tf})
    filepaths = DirUtils.list_dir_full_path(args.input_path, interest_extensions=[".nii.gz", ".gz"])
    os.makedirs(args.output_path, exist_ok=True)

    for file_path in filepaths:
        file_name = os.path.split(file_path)[-1]
        img = nib.load(file_path)
        # loadtest = np.array(img.get_fdata())
        # print("Original:", loadtest.shape)
        hdr = img.header
        affinemat = img.affine
        # loadtest = np.swapaxes(loadtest, 0, 2)
        # loadtest = np.expand_dims(loadtest, axis=3)
        loadtest = load_nii_file(file_path)
        print("data shape:", loadtest.shape)

        segmentation_array = np.zeros((loadtest.shape[2], loadtest.shape[1], loadtest.shape[0]))
        print("Seg Array:", segmentation_array.shape)
        prediction_start_index = 0
        while prediction_start_index < loadtest.shape[0]:
            prediction_end_index = prediction_start_index + pred_batch_size
            if prediction_end_index > loadtest.shape[0]:
                prediction_end_index = loadtest.shape[0]
            data = loadtest[prediction_start_index:prediction_end_index, ...]
            data = data / 255 # Normalize the inputs
            pred = model.predict(data)
            pred = np.array(pred)
            pred = pred[0, :, :, :, 0]
            pred = np.swapaxes(pred, 0, 2)
            segmentation_array[:, :, prediction_start_index:prediction_end_index] = pred
            prediction_start_index = prediction_end_index
        segmentation_array = segmentation_array > 0.5  # Set to one
        segmentation_array = np.asarray(segmentation_array)
        new_img = nib.nifti1.Nifti1Image(segmentation_array, affinemat, hdr)
        print(30 * '___')
        nib.save(new_img, os.path.join(args.output_path, file_name))
