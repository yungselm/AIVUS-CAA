import numpy as np
import tensorflow.keras.backend as K



def dice_score_tf(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_score_np(y_true: np.ndarray, y_pred: np.ndarray, smooth=1):
    y_true_f = y_true.astype(np.int8).flatten()
    y_pred_f = y_pred.astype(np.int8).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def monai_dice(y_true: np.ndarray, y_pred: np.ndarray):
    from monai.metrics import DiceMetric, compute_dice
    metric =  compute_dice(y_pred, y_true, num_classes=1)
    print(metric)

if __name__ == '__main__':
    from deep_utils import NIBUtils
    y_pred = "/home/aicvi/projects/NIVA/segmentation_train/output/last_new_masks_ts/PDWR8LS1_frame_22_0000.nii.gz"
    y_pred = NIBUtils.get_array(y_pred).astype(np.int8)
    y_true = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset007_Ivus/labelsTs/PDWR8LS1_frame_22.nii.gz"
    y_true = NIBUtils.get_array(y_true).astype(np.int8)

    print(np.unique(y_true), np.unique(y_pred))
    print(dice_score_np(y_true, y_pred))
    # monai_dice()