import numpy as np

from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QInputDialog,
    QLineEdit,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt


class MetadataWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.table = main_window.metadata_table
        self.setWindowTitle('Metadata')

        self.fitToTable()
        self.setCentralWidget(self.table)

    def fitToTable(self):
        x = sum([self.table.columnWidth(i) for i in range(self.table.columnCount())])
        y = sum([self.table.rowHeight(i) for i in range(self.table.rowCount())])
        self.setFixedSize(x, y)

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
        gender = main_window.dicom.PatientSex
    else:
        gender = 'Unknown'

    if main_window.dicom.get('IVUSPullbackRate'):
        pullback_rate = float(main_window.dicom.IVUSPullbackRate)
    # check Boston private tag
    elif main_window.dicom.get(0x000B1001):
        pullback_rate = float(main_window.dicom[0x000B1001].value)
    else:
        pullback_rate, _ = QInputDialog.getText(
            main_window,
            'Pullback Speed',
            'No pullback speed found, please enter pullback speed (mm/s)',
            QLineEdit.Normal,
            '0.5',
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
            'Pixel Spacing',
            'No pixel spacing info found, please enter pixel spacing (mm)',
            QLineEdit.Normal,
            '',
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

    main_window.metadata_table.setRowCount(8)
    main_window.metadata_table.setColumnCount(2)
    main_window.metadata_table.setItem(0, 0, QTableWidgetItem('Patient Name'))
    main_window.metadata_table.setItem(0, 1, QTableWidgetItem(patient_name))
    main_window.metadata_table.setItem(1, 0, QTableWidgetItem('Date of Birth'))
    main_window.metadata_table.setItem(1, 1, QTableWidgetItem(birth_date))
    main_window.metadata_table.setItem(2, 0, QTableWidgetItem('Gender'))
    main_window.metadata_table.setItem(2, 1, QTableWidgetItem(gender))
    main_window.metadata_table.setItem(3, 0, QTableWidgetItem('Pullback Speed'))
    main_window.metadata_table.setItem(3, 1, QTableWidgetItem(str(pullback_rate)))
    main_window.metadata_table.setItem(4, 0, QTableWidgetItem('Resolution (mm)'))
    main_window.metadata_table.setItem(4, 1, QTableWidgetItem(str(main_window.metadata['resolution'])))
    main_window.metadata_table.setItem(5, 0, QTableWidgetItem('Dimensions'))
    main_window.metadata_table.setItem(5, 1, QTableWidgetItem(str(rows)))
    main_window.metadata_table.setItem(6, 0, QTableWidgetItem('Manufacturer'))
    main_window.metadata_table.setItem(6, 1, QTableWidgetItem(manufacturer))
    main_window.metadata_table.setItem(7, 0, QTableWidgetItem('Model'))
    main_window.metadata_table.setItem(7, 1, QTableWidgetItem((model)))

    main_window.metadata_table.horizontalHeader().hide()
    main_window.metadata_table.verticalHeader().hide()
    main_window.metadata_table.resizeColumnsToContents()
    main_window.metadata_table.resizeRowsToContents()
    main_window.metadata_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    main_window.metadata_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
