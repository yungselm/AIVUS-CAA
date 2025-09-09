from sklearn.metrics import confusion_matrix
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


def specificity_and_sensitivity(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return specificity, sensitivity
