import os

import pydicom as dcm
import SimpleITK as sitk
import numpy as np
from loguru import logger
from PyQt5.QtWidgets import (
    QInputDialog,
    QMessageBox,
    QLineEdit,
    QFileDialog,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt

from input_output.contours_io import read_contours


def read_image(main_window):
    """
    Reads DICOM or NIfTi images.

    Reads the DICOM/NIfTi images and metadata. Places metatdata in a table.
    Images are displayed in the graphics scene.
    """
    main_window.status_bar.showMessage('Reading image file...')
    options = QFileDialog.Options()
    options = QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(
        main_window, "QFileDialog.getOpenFileName()", "", "All files (*)", options=options
    )

    if file_name:
        main_window.file_name = os.path.splitext(file_name)[0]  # remove file extension
        try:  # DICOM
            main_window.dicom = dcm.read_file(file_name, force=True)
            main_window.images = main_window.dicom.pixel_array
            parse_dicom(main_window)
        except AttributeError:
            try:  # NIfTi
                main_window.images = sitk.GetArrayFromImage(sitk.ReadImage(file_name))
                main_window.file_name = main_window.file_name.split('_')[0]  # remove _img.nii suffix
            except:
                error = QMessageBox()
                error.setIcon(QMessageBox.Critical)
                error.setWindowTitle("Error")
                error.setModal(True)
                error.setWindowModality(Qt.WindowModal)
                error.setText("File is not a valid IVUS file and could not be loaded (DICOM or NIfTi supported)")
                error.exec_()
                return None

        main_window.metadata['num_frames'] = main_window.images.shape[0]
        main_window.display_slider.setMaximum(main_window.metadata['num_frames'] - 1)

        success = read_contours(main_window, main_window.file_name)
        if success:
            main_window.segmentation = True
            try:
                main_window.gated_frames_dia = [
                    frame
                    for frame in range(main_window.metadata['num_frames'])
                    if main_window.data['phases'][frame] == 'D'
                ]
                main_window.gated_frames_sys = [
                    frame
                    for frame in range(main_window.metadata['num_frames'])
                    if main_window.data['phases'][frame] == 'S'
                ]
                main_window.display_slider.set_gated_frames(main_window.gated_frames_dia)
            except KeyError:  # old contour files may not have phases attribute
                pass
        else:  # initialise empty containers
            for key in ['plaque_frames', 'lumen_area', 'lumen_circumf', 'longest_distance', 'shortest_distance']:
                main_window.data[key] = [0] * main_window.metadata['num_frames']
            main_window.data['phases'] = ['-'] * main_window.metadata['num_frames']
            for key in ['lumen_centroid', 'farthest_point', 'nearest_point', 'lumen']:
                main_window.data[key] = (
                [[] for _ in range(main_window.metadata['num_frames'])],
                [[] for _ in range(main_window.metadata['num_frames'])],
            )
            main_window.display.set_data(main_window.data['lumen'], main_window.images)

        main_window.image_displayed = True
        main_window.display_slider.setValue(main_window.metadata['num_frames'] - 1)
    main_window.status_bar.showMessage('Waiting for user input')


def parse_dicom(main_window):
    """Parses DICOM metadata"""
    if len(main_window.dicom.PatientName.encode('ascii')) > 0:
        patient_name = main_window.dicom.PatientName.original_string.decode('utf-8')
    else:
        patient_name = 'Unknown'

    if len(main_window.dicom.PatientBirthDate) > 0:
        birth_date = main_window.dicom.PatientBirthDate
    else:
        birth_date = 'Unknown'

    if len(main_window.dicom.PatientSex) > 0:
        sex = main_window.dicom.PatientSex
    else:
        sex = 'Unknown'

    if main_window.dicom.get('IVUSPullbackRate'):
        pullback_rate = float(main_window.dicom.IVUSPullbackRate)
    # check Boston private tag
    elif main_window.dicom.get(0x000B1001):
        pullback_rate = float(main_window.dicom[0x000B1001].value)
    else:
        pullback_rate, _ = QInputDialog.getText(
            main_window,
            "Pullback Speed",
            "No pullback speed found, please enter pullback speeed (mm/s)",
            QLineEdit.Normal,
            "0.5",
        )
        pullback_rate = float(pullback_rate)

    if main_window.dicom.get('FrameTimeVector'):
        frame_time_vector = main_window.dicom.get('FrameTimeVector')
        frame_time_vector = [float(frame) for frame in frame_time_vector]
        pullback_time = np.cumsum(frame_time_vector) / 1000  # assume in ms
        pullback_length = pullback_time * float(pullback_rate)
    else:
        pullback_length = np.zeros((main_window.images.shape[0],))

    main_window.metadata['pullback_length'] = pullback_length

    if main_window.dicom.get('SequenceOfUltrasoundRegions'):
        if main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalUnitsXDirection == 3:
            # pixels are in cm, convert to mm
            resolution = main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX * 10
        else:
            # assume mm
            resolution = main_window.dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    elif main_window.dicom.get('PixelSpacing'):
        resolution = float(main_window.dicom.PixelSpacing[0])
    else:
        resolution, _ = QInputDialog.getText(
            main_window,
            "Pixel Spacing",
            "No pixel spacing info found, please enter pixel spacing (mm)",
            QLineEdit.Normal,
            "",
        )
        resolution = float(resolution)

    main_window.metadata['resolution'] = resolution

    if main_window.dicom.get('Rows'):
        rows = main_window.dicom.Rows
    else:
        rows = main_window.images.shape[1]

    if main_window.dicom.get('Manufacturer'):
        manufacturer = main_window.dicom.Manufacturer
    else:
        manufacturer = 'Unknown'

    if main_window.dicom.get('ManufacturerModelName'):
        model = main_window.dicom.ManufacturerModelName
    else:
        model = 'Unknown'

    main_window.info_table.setItem(0, 1, QTableWidgetItem(patient_name))
    main_window.info_table.setItem(1, 1, QTableWidgetItem(birth_date))
    main_window.info_table.setItem(2, 1, QTableWidgetItem(sex))
    main_window.info_table.setItem(3, 1, QTableWidgetItem(str(pullback_rate)))
    main_window.info_table.setItem(4, 1, QTableWidgetItem(str(main_window.metadata['resolution'])))
    main_window.info_table.setItem(5, 1, QTableWidgetItem(str(rows)))
    main_window.info_table.setItem(6, 1, QTableWidgetItem(manufacturer))
    main_window.info_table.setItem(7, 1, QTableWidgetItem((model)))
