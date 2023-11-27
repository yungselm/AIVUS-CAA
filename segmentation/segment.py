import numpy as np
import matplotlib.path as mplPath
from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtCore import Qt
from skimage import measure


def segment(main_window):
    """Automatic segmentation of IVUS images"""
    main_window.status_bar.showMessage('Segmenting all gated frames...')
    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot perform automatic segmentation before reading DICOM file')
        warning.exec_()
        main_window.status_bar.showMessage('Waiting for user input')
        return

    masks = main_window.predictor(main_window.images)
    main_window.data['lumen'] = mask_to_contours(masks)
    main_window.contours_drawn = True
    main_window.display.set_data(main_window.data['lumen'], main_window.images)
    main_window.hide_contours_box.setChecked(False)
    main_window.status_bar.showMessage('Waiting for user input')


def mask_to_contours(masks):
    """Convert numpy mask to IVUS contours"""
    lumen_pred = get_contours(masks, image_shape=masks.shape[1:3])
    lumen_pred = downsample(lumen_pred)

    return lumen_pred


def get_contours(preds, image_shape):
    """Extracts contours from masked images. Returns x and y coodinates"""
    lumen_pred = [[], []]
    for frame in range(preds.shape[0]):
        if np.any(preds[frame, :, :] == 1):
            lumen = label_contours(preds[frame, :, :])
            keep_lumen_x, keep_lumen_y = keep_largest_contour(lumen, image_shape)
            lumen_pred[0].append(keep_lumen_x)
            lumen_pred[1].append(keep_lumen_y)
        else:
            lumen_pred[0].append([])
            lumen_pred[1].append([])

    return lumen_pred


def label_contours(image):
    """generate contours for labels"""
    contours = measure.find_contours(image)
    lumen = []
    for contour in contours:
        lumen.append(np.array((contour[:, 0], contour[:, 1])))

    return lumen


def keep_largest_contour(contours, image_shape):
    max_length = 0
    keep_contour = [[], []]
    for contour in contours:
        if keep_valid_contour(contour, image_shape):
            if len(contour[0]) > max_length:
                keep_contour = [list(contour[1, :]), list(contour[0, :])]
                max_length = len(contour[0])

    return keep_contour


def keep_valid_contour(contour, image_shape):
    """Contour is valid if it contains the centroid of the image"""
    bbPath = mplPath.Path(np.transpose(contour))
    centroid = [image_shape[0] // 2, image_shape[1] // 2]
    return bbPath.contains_point(centroid)


def downsample(contours, num_points=20):
    """Downsamples input contour data by selecting n points from original contour"""
    num_frames = len(contours[0])
    downsampled = [[] for _ in range(num_frames)], [[] for _ in range(num_frames)]

    for frame in range(num_frames):
        if contours[0][frame]:
            points_to_sample = range(0, len(contours[0][frame]), len(contours[0][frame]) // num_points)
            for axis in range(2):
                downsampled[axis][frame] = [contours[axis][frame][point] for point in points_to_sample]
    return downsampled
