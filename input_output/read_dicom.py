import os

import pydicom as dcm
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

from input_output.contours import readContours


def readDICOM(main_window):
    """Reads DICOM images.

    Reads the dicom images and metadata. Places metatdata in a table.
    Images are displayed in the graphics scene.
    """
    main_window.status_bar.showMessage('Reading DICOM file...')
    options = QFileDialog.Options()
    options = QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(
        main_window, "QFileDialog.getOpenFileName()", "", "All files (*)", options=options
    )

    if fileName:
        try:
            main_window.dicom = dcm.read_file(fileName, force=True)
            main_window.images = main_window.dicom.pixel_array
            main_window.file_name = os.path.splitext(fileName)[0]  # remove file extension
        except:
            error = QMessageBox()
            error.setIcon(QMessageBox.Critical)
            error.setWindowTitle("Error")
            error.setModal(True)
            error.setWindowModality(Qt.WindowModal)
            error.setText("File is not a valid IVUS file and could not be loaded")
            error.exec_()
            return None

        main_window.metadata['number_of_frames'] = int(main_window.dicom.NumberOfFrames)
        main_window.slider.setMaximum(main_window.metadata['number_of_frames'] - 1)
        parseDICOM(main_window)

        try:
            success = readContours(main_window, main_window.file_name)
            main_window.segmentation = True
            try:
                main_window.gated_frames_dia = [
                    frame
                    for frame in range(main_window.metadata['number_of_frames'])
                    if main_window.data['phases'][frame] == 'D'
                ]
                main_window.gated_frames_sys = [
                    frame
                    for frame in range(main_window.metadata['number_of_frames'])
                    if main_window.data['phases'][frame] == 'S'
                ]
            except IndexError:  # old contour files may not have phases attr
                pass
            main_window.gated_frames = main_window.gated_frames_dia
            main_window.slider.addGatedFrames(main_window.gated_frames)
        except FileNotFoundError:
            main_window.data['plaque_frames'] = ['0'] * main_window.metadata['number_of_frames']
            main_window.data['phases'] = ['-'] * main_window.metadata['number_of_frames']

            (
                main_window.data['lumen_area'],
                main_window.data['lumen_circumf'],
                main_window.data['longest_distance'],
                main_window.data['shortest_distance'],
            ) = [[0] * main_window.metadata['number_of_frames'] for _ in range(4)]
            (  # initialise empty containers
                main_window.data['lumen_centroid'],
                main_window.data['farthest_point'],
                main_window.data['nearest_point'],
                main_window.data['lumen'],
            ) = [
                (
                    [[] for _ in range(main_window.metadata['number_of_frames'])],
                    [[] for _ in range(main_window.metadata['number_of_frames'])],
                )
                for _ in range(4)
            ]

        if not success:  # else data already set
            main_window.display.setData(main_window.data['lumen'], main_window.images)
        main_window.image_displayed = True
        main_window.slider.setValue(main_window.metadata['number_of_frames'] - 1)
    main_window.status_bar.showMessage('Waiting for user input')


def parseDICOM(main_window):
    """Parses DICOM metadata"""
    if len(main_window.dicom.PatientName.encode('ascii')) > 0:
        patientName = main_window.dicom.PatientName.original_string.decode('utf-8')
    else:
        patientName = 'Unknown'

    if len(main_window.dicom.PatientBirthDate) > 0:
        patientBirthDate = main_window.dicom.PatientBirthDate
    else:
        patientBirthDate = 'Unknown'

    if len(main_window.dicom.PatientSex) > 0:
        patientSex = main_window.dicom.PatientSex
    else:
        patientSex = 'Unknown'

    if main_window.dicom.get('IVUSPullbackRate'):
        ivusPullbackRate = float(main_window.dicom.IVUSPullbackRate)
    # check Boston private tag
    elif main_window.dicom.get(0x000B1001):
        ivusPullbackRate = float(main_window.dicom[0x000B1001].value)
    else:
        ivusPullbackRate, _ = QInputDialog.getText(
            main_window,
            "Pullback Speed",
            "No pullback speed found, please enter pullback speeed (mm/s)",
            QLineEdit.Normal,
            "0.5",
        )
        ivusPullbackRate = float(ivusPullbackRate)

    if main_window.dicom.get('FrameTimeVector'):
        frameTimeVector = main_window.dicom.get('FrameTimeVector')
        frameTimeVector = [float(frame) for frame in frameTimeVector]
        pullbackTime = np.cumsum(frameTimeVector) / 1000  # assume in ms
        pullbackLength = pullbackTime * float(ivusPullbackRate)
    else:
        pullbackLength = np.zeros((main_window.images.shape[0],))

    main_window.metadata['pullback_length'] = pullbackLength

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

    main_window.infoTable.setItem(0, 1, QTableWidgetItem(patientName))
    main_window.infoTable.setItem(1, 1, QTableWidgetItem(patientBirthDate))
    main_window.infoTable.setItem(2, 1, QTableWidgetItem(patientSex))
    main_window.infoTable.setItem(3, 1, QTableWidgetItem(str(ivusPullbackRate)))
    main_window.infoTable.setItem(4, 1, QTableWidgetItem(str(main_window.metadata['resolution'])))
    main_window.infoTable.setItem(5, 1, QTableWidgetItem(str(rows)))
    main_window.infoTable.setItem(6, 1, QTableWidgetItem(manufacturer))
    main_window.infoTable.setItem(7, 1, QTableWidgetItem((model)))
