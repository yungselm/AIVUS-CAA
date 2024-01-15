import numpy as np
import matplotlib.path as mplPath
from loguru import logger
from skimage import measure

from gui.error_message import ErrorMessage
from gui.frame_range_dialog import FrameRangeDialog


def segment(main_window):
    """Automatic segmentation of IVUS images"""
    main_window.status_bar.showMessage('Segmenting frames...')
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot perform automatic segmentation before reading DICOM file')
        main_window.status_bar.showMessage(main_window.waiting_status)
        return

    segment_dialog = FrameRangeDialog(main_window)

    if segment_dialog.exec_():
        lower_limit, upper_limit = segment_dialog.getInputs()
        masks = main_window.predictor(main_window.images, lower_limit, upper_limit)
        if masks is not None:
            main_window.data['lumen'] = mask_to_contours(masks, main_window.config.display.n_interactive_points)
            main_window.data['lumen_area'] = [0] * main_window.metadata[
                'num_frames'
            ]  # ensure all metrics are recalculated for the report
            main_window.contours_drawn = True
            main_window.display.set_data(main_window.data['lumen'], main_window.images)
            main_window.hide_contours_box.setChecked(False)

    main_window.status_bar.showMessage(main_window.waiting_status)


def mask_to_contours(masks, num_points):
    """Convert numpy mask to IVUS contours"""
    lumen_pred = get_contours(masks, image_shape=masks.shape[1:3])
    lumen_pred = downsample(lumen_pred, num_points)

    return lumen_pred


def get_contours(preds, image_shape):
    """Extracts contours from masked images. Returns x and y coodinates"""
    lumen_pred = [[], []]
    counter = 0
    for frame in range(preds.shape[0]):
        if np.sum(preds[frame, :, :]) > 0:
            counter += 1
            lumen = label_contours(preds[frame, :, :])
            keep_lumen_x, keep_lumen_y = keep_largest_contour(lumen, image_shape)
            lumen_pred[0].append(keep_lumen_x)
            lumen_pred[1].append(keep_lumen_y)
        else:
            lumen_pred[0].append([])
            lumen_pred[1].append([])
    logger.info(f'Found contours in {counter} frames')
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


def downsample(contours, num_points):
    """Downsamples input contour data by selecting n points from original contour"""
    num_frames = len(contours[0])
    downsampled = [[] for _ in range(num_frames)], [[] for _ in range(num_frames)]

    for frame in range(num_frames):
        if contours[0][frame]:
            points_to_sample = range(0, len(contours[0][frame]), len(contours[0][frame]) // num_points)
            for axis in range(2):
                downsampled[axis][frame] = [contours[axis][frame][point] for point in points_to_sample]
    return downsampled
