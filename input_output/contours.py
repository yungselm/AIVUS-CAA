import os
import csv

import numpy as np
from loguru import logger
from skimage.draw import polygon2mask
from PyQt5.QtWidgets import (
    QErrorMessage,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from input_output.read_xml import read
from input_output.write_xml import write_xml, mask_image, get_contours


def readContours(window, fileName=None):
    """Reads contours.

    Reads contours  saved in xml format (Echoplaque compatible) and
    displays the contours in the graphics scene
    """

    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Reading of contours failed. Images must be loaded prior to loading contours')
        warning.exec_()
        return

    if fileName is None:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            window, "QFileDialog.getOpenFileName()", "", "XML file (*.xml)", options=options
        )

    if fileName:
        window.lumen, window.resolution, _, window.plaque_frames, window.phases = read(fileName)

        window.resolution = float(window.resolution[0])
        window.lumen = mapToList(window.lumen)
        window.contours = True
        window.wid.setData(window.lumen, window.images)
        window.hideBox.setChecked(False)


def writeContours(window):
    """Writes contours to an xml file compatible with Echoplaque"""

    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot write contours before reading DICOM file')
        warning.exec_()
        return

    window.lumen = window.wid.getData()

    # write contours to .csv file
    csv_out_dir = os.path.join(window.file_name + '_csv_files')
    os.makedirs(csv_out_dir, exist_ok=True)
    contoured_frames = [
        frame for frame in range(window.numberOfFrames) if window.lumen[0][frame]
    ]  # find frames with contours (no need to save the others)

    for frame in contoured_frames:
        with open(os.path.join(csv_out_dir, f'{frame}_contours.csv'), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            rows = zip(window.lumen[0][frame], window.lumen[1][frame])  # csv can only write rows, not columns directly
            for row in rows:
                writer.writerow(row)

    # reformat data for compatibility with write_xml function
    x, y = [], []
    for i in range(window.numberOfFrames):
        if i < len(window.lumen[0]):
            new_x_lumen = window.lumen[0][i]
            new_y_lumen = window.lumen[1][i]
        else:
            new_x_lumen = []
            new_y_lumen = []

        x.append(new_x_lumen)
        y.append(new_y_lumen)

    write_xml(
        x,
        y,
        window.images.shape,
        window.resolution,
        window.ivusPullbackRate,
        window.plaque_frames,
        window.phases,
        window.file_name,
    )


def reset_contours(window):
    window.contours = False
    window.lumen = None


def segment(window):
    """Segmentation and phenotyping of IVUS images"""
    window.status_bar.showMessage('Segmenting all gated frames...')
    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot perform automatic segmentation before reading DICOM file')
        warning.exec_()
        window.status_bar.showMessage('Waiting for user input')
        return

    save_path = os.path.join(os.getcwd(), "model", "saved_model.pb")
    if not os.path.isfile(save_path):
        message = "No saved weights have been found, check that weights are saved in {}".format(
            os.path.join(os.getcwd(), "model")
        )
        error = QMessageBox()
        error.setIcon(QMessageBox.Critical)
        error.setWindowTitle("Error")
        error.setModal(True)
        error.setWindowModality(Qt.WindowModal)
        error.setText(message)
        error.exec_()
        window.status_bar.showMessage('Waiting for user input')
        return -1

    image_dim = window.images.shape

    if hasattr(window, 'masks'):  # keep previous segmentation
        masks = window.masks
    else:  # perform first segmentation
        masks = np.zeros((window.numberOfFrames, image_dim[1], image_dim[2]), dtype=np.uint8)

    # masks[window.gated_frames, :, :] = predict(window.images[window.gated_frames, :, :])
    window.masks = masks

    # compute metrics such as plaque burden
    window.metrics = computeMetrics(window, masks)

    # convert masks to contours
    window.lumen = maskToContours(masks)
    window.contours = True

    window.wid.setData(window.lumen, window.images)
    window.hideBox.setChecked(False)
    window.status_bar.showMessage('Waiting for user input')


def newSpline(window):
    """Create a message box to choose what spline to create"""

    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot create manual contour before reading DICOM file')
        warning.exec_()
        return

    window.wid.new(window)
    window.hideBox.setChecked(False)
    window.contours = True


def maskToContours(masks):
    """Convert numpy mask to IVUS contours"""

    levels = [1.5, 2.5]
    image_shape = masks.shape[1:3]
    masks = mask_image(masks, catheter=0)
    _, _, lumen_pred = get_contours(masks, levels, image_shape)

    return lumen_pred


def contoursToMask(images, contoured_frames, lumen):
    """Convert IVUS contours to numpy mask"""
    image_shape = images.shape[1:3]
    mask = np.zeros_like(images)
    for i, frame in enumerate(contoured_frames):
        try:
            lumen_polygon = [[x, y] for x, y in zip(lumen[1][frame], lumen[0][frame])]
            mask[i, :, :] += polygon2mask(image_shape, lumen_polygon).astype(np.uint8)
        except ValueError:  # frame has no lumen contours
            pass
    mask = np.clip(mask, a_min=0, a_max=1)  # enforce correct value range

    return mask


def computeMetrics(window, masks):
    """Measures lumen area"""
    lumen_area = np.sum(masks == 1, axis=(1, 2)) * window.resolution**2

    return lumen_area


def mapToList(contours):
    """Converts map to list"""
    x, y = contours
    x = [list(x[i]) for i in range(0, len(x))]
    y = [list(y[i]) for i in range(0, len(y))]

    return (x, y)
