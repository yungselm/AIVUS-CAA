import os
import csv
import json
import glob

import numpy as np
from loguru import logger
from skimage.draw import polygon2mask
from PyQt5.QtWidgets import (
    QErrorMessage,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from version import version_file_str
from input_output.read_xml import read
from input_output.write_xml import write_xml, mask_image, get_contours


def readContours(main_window, file_name=None):
    """Reads contours saved in json/xml format and displays the contours in the graphics scene"""
    success = False
    json_files = glob.glob(f'{file_name}_contours*.json')
    xml_files = glob.glob(f'{file_name}_contours*.xml')

    if file_name is None:  # call by manual button click
        if not main_window.image_displayed:
            warning = QErrorMessage(main_window)
            warning.setWindowModality(Qt.WindowModal)
            warning.showMessage('Reading of contours failed. Images must be loaded prior to loading contours')
            warning.exec_()
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            main_window, "QFileDialog.getOpenFileName()", "", "All files (*)", options=options
        )

        if os.path.splitext(file_name)[1] == '.json':
            with open(file_name, 'r') as in_file:
                main_window.data = json.load(in_file)
            success = True
        elif os.path.splitext(file_name)[1] == '.xml':
            (
                main_window.data['lumen'],
                main_window.metadata['resolution'],
                _,
                main_window.data['plaque_frames'],
                main_window.data['phases'],
            ) = read(file_name)
            main_window.metadata['resolution'] = float(main_window.metadata['resolution'][0])
            main_window.data['lumen'] = mapToList(main_window.data['lumen'])
            success = True

    elif not main_window.use_xml_files and json_files:  # json files have priority over xml unless desired
        newest_json = max(json_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_json}')
        with open(newest_json, 'r') as in_file:
            main_window.data = json.load(in_file)
        success = True

    elif xml_files:
        newest_xml = max(xml_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_xml}')
        (
            main_window.data['lumen'],
            main_window.metadata['resolution'],
            _,
            main_window.data['plaque_frames'],
            main_window.data['phases'],
        ) = read(newest_xml)
        main_window.metadata['resolution'] = float(main_window.metadata['resolution'][0])
        main_window.data['lumen'] = mapToList(main_window.data['lumen'])
        (  # initialise empty containers
            main_window.data['lumen_centroid'],
            main_window.data['farthest_point'],
            main_window.data['nearest_point'],
        ) = [
            (
                [[] for _ in range(main_window.metadata['number_of_frames'])],
                [[] for _ in range(main_window.metadata['number_of_frames'])],
            )
            for _ in range(3)
        ]
        (
            main_window.data['lumen_area'],
            main_window.data['longest_distance'],
            main_window.data['shortest_distance'],
        ) = [[0] * main_window.metadata['number_of_frames'] for _ in range(3)]
        success = True

    if success:
        main_window.contours_drawn = True
        main_window.display.setData(main_window.data['lumen'], main_window.images)
        main_window.hideBox.setChecked(False)

    return success


def writeContours(main_window):
    """Writes contours to a json/xml file"""

    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot write contours before reading DICOM file')
        warning.exec_()
        return

    if main_window.use_xml_files:
        # reformat data for compatibility with write_xml function
        x, y = [], []
        for i in range(main_window.metadata['number_of_frames']):
            if i < len(main_window.data['lumen'][0]):
                new_x_lumen = main_window.data['lumen'][0][i]
                new_y_lumen = main_window.data['lumen'][1][i]
            else:
                new_x_lumen = []
                new_y_lumen = []

            x.append(new_x_lumen)
            y.append(new_y_lumen)

        write_xml(
            x,
            y,
            main_window.images.shape,
            main_window.metadata['resolution'],
            main_window.ivusPullbackRate,
            main_window.data['plaque_frames'],
            main_window.data['phases'],
            main_window.file_name,
        )
    else:
        with open(os.path.join(main_window.file_name + f'_contours_{version_file_str}.json'), 'w') as out_file:
            json.dump(main_window.data, out_file)


def reset_contours(main_window):
    main_window.contours_drawn = False
    main_window.data['lumen'] = None


def segment(main_window):
    """Segmentation and phenotyping of IVUS images"""
    main_window.status_bar.showMessage('Segmenting all gated frames...')
    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot perform automatic segmentation before reading DICOM file')
        warning.exec_()
        main_window.status_bar.showMessage('Waiting for user input')
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
        main_window.status_bar.showMessage('Waiting for user input')
        return -1

    image_dim = main_window.images.shape

    if hasattr(main_window, 'masks'):  # keep previous segmentation
        masks = main_window.masks
    else:  # perform first segmentation
        masks = np.zeros((main_window.metadata['number_of_frames'], image_dim[1], image_dim[2]), dtype=np.uint8)

    # masks[main_window.gated_frames, :, :] = predict(main_window.images[main_window.gated_frames, :, :])
    main_window.masks = masks

    # compute metrics such as plaque burden
    main_window.metrics = computeMetrics(main_window, masks)

    # convert masks to contours
    main_window.data['lumen'] = maskToContours(masks)
    main_window.contours_drawn = True

    main_window.display.setData(main_window.data['lumen'], main_window.images)
    main_window.hideBox.setChecked(False)
    main_window.status_bar.showMessage('Waiting for user input')


def newSpline(main_window):
    """Create a message box to choose what spline to create"""

    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot create manual contour before reading DICOM file')
        warning.exec_()
        return

    main_window.display.new(main_window)
    main_window.hideBox.setChecked(False)
    main_window.contours_drawn = True


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


def computeMetrics(main_window, masks):
    """Measures lumen area"""
    lumen_area = np.sum(masks == 1, axis=(1, 2)) * main_window.metadata['resolution'] ** 2

    return lumen_area


def mapToList(contours):
    """Converts map to list"""
    x, y = contours
    x = [list(x[i]) for i in range(0, len(x))]
    y = [list(y[i]) for i in range(0, len(y))]

    return (x, y)
