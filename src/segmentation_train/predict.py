import pandas as pd
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from deep_utils import DirUtils, NIBUtils
from tensorflow.keras.models import load_model

from configs import *
from metrics import dice_score_tf, dice_score_np, specificity_and_sensitivity


def load_nii_file(fpath):
    arr = nib.load(fpath)
    arr = np.asanyarray(arr.dataobj)
    if len(arr.shape) == 2:
        arr = arr[..., None]
    arr = np.swapaxes(arr, 0, 2)
    return arr


exp_number = 52
suffix = BEST_SUFFIX  # LAST_SUFFIX
parser = ArgumentParser()

parser.add_argument("--input_path", default="output/images", help="path to the images")
parser.add_argument("--output_path", default="output/masks", help="where predicted masks will be saved")
parser.add_argument("--true_masks", default=None,
                    help="Where true masks are saved to return the dice excel")
parser.add_argument("--model_path", default=f"models/exp_{exp_number}/{TRIAL_IDENTIFIER}{suffix}.h5")

args = parser.parse_args()

pred_batch_size = 256

if __name__ == '__main__':
    model = load_model(args.model_path, custom_objects={"dice_score_tf": dice_score_tf})
    filepaths = DirUtils.list_dir_full_path(args.input_path, interest_extensions=[".nii.gz", ".gz"])
    os.makedirs(args.output_path, exist_ok=True)
    df = []
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

        prediction_start_index = 0
        while prediction_start_index < loadtest.shape[0]:
            prediction_end_index = prediction_start_index + pred_batch_size
            if prediction_end_index > loadtest.shape[0]:
                prediction_end_index = loadtest.shape[0]
            data = loadtest[prediction_start_index:prediction_end_index, ...]
            data = data / 255  # Normalize the inputs
            pred = model.predict(data)
            pred = np.array(pred)
            pred = pred[0, :, :, :, 0]
            pred = np.swapaxes(pred, 0, 2)
            segmentation_array[:, :, prediction_start_index:prediction_end_index] = pred
            prediction_start_index = prediction_end_index
        segmentation_array[segmentation_array > 0.5] = 1  # Set to one
        segmentation_array[segmentation_array <= 0.5] = 0  # Set to zero
        segmentation_array = np.asarray(segmentation_array)
        new_img = nib.nifti1.Nifti1Image(segmentation_array, affinemat, hdr)
        print("Seg Array:", segmentation_array.shape)
        if args.true_masks is not None:
            true_mask_path = join(args.true_masks, file_name.replace('_0000', ''))
            true_mask = NIBUtils.get_array(true_mask_path)
            print("true_mask Array:", true_mask.shape)
            dice = dice_score_np(true_mask, segmentation_array[..., 0])
            specificity, sensitivity = specificity_and_sensitivity(true_mask, segmentation_array[..., 0])
        else:
            dice = "-"
            specificity = "-"
            sensitivity = "-"
        df.append([file_name, dice, specificity, sensitivity])
        print(30 * '___')
        nib.save(new_img, os.path.join(args.output_path, file_name))
    df = pd.DataFrame(df, columns=['filename', 'dice', "specificity", "sensitivity"])
    df.to_csv(args.output_path + ".csv")
    print("Mean Dice", df['dice'].mean())
    print("STD Dice", df['dice'].std())
    print("Mean sensitivity", df['sensitivity'].mean())
    print("STD sensitivity", df['sensitivity'].std())
    print("Mean specificity", df['specificity'].mean())
    print("STD specificity", df['specificity'].std())
