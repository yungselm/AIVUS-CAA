import os
import json
import glob

from loguru import logger
from PyQt5.QtWidgets import (
    QErrorMessage,
)
from PyQt5.QtCore import Qt

from version import version_file_str
from input_output.read_xml import read_xml
from input_output.write_xml import write_xml


def read_contours(main_window, file_name=None):
    """Reads contours saved in json/xml format and displays the contours in the graphics scene"""
    success = False
    json_files = glob.glob(f'{file_name}_contours*.json')
    xml_files = glob.glob(f'{file_name}_contours*.xml')

    if not main_window.use_xml_files and json_files:  # json files have priority over xml unless desired
        newest_json = max(json_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_json}')
        with open(newest_json, 'r') as in_file:
            main_window.data = json.load(in_file)
        success = True

    elif xml_files:
        newest_xml = max(xml_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_xml}')
        read_xml(main_window, newest_xml)
        main_window.data['lumen'] = map_to_list(main_window.data['lumen'])
        for key in ['lumen_area', 'lumen_circumf', 'longest_distance', 'shortest_distance']:
            main_window.data[key] = [0] * main_window.metadata[
                'num_frames'
            ]  # initialise empty containers for data not stored in xml
        for key in ['lumen_centroid', 'farthest_point', 'nearest_point']:
            main_window.data[key] = (
                [[] for _ in range(main_window.metadata['num_frames'])],
                [[] for _ in range(main_window.metadata['num_frames'])],
            )  # initialise empty containers for data not stored in xml
        success = True

    if success:
        main_window.contours_drawn = True
        main_window.display.set_data(main_window.data['lumen'], main_window.images)
        main_window.hide_contours_box.setChecked(False)

    return success


def write_contours(main_window):
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
        for frame in range(main_window.metadata['num_frames']):
            if frame < len(main_window.data['lumen'][0]):
                new_x_lumen = main_window.data['lumen'][0][frame]
                new_y_lumen = main_window.data['lumen'][1][frame]
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


def map_to_list(contours):
    """Converts map to list"""
    x, y = contours
    x = [list(x[i]) for i in range(len(x))]
    y = [list(y[i]) for i in range(len(y))]

    return (x, y)
