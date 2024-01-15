import time
import bisect

from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHeaderView,
    QStyle,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from gui.display.display import Display
from gui.slider import Slider, Communicate
from gui.shortcuts import init_shortcuts
from gui.display.contours_gui import new_contour
from input_output.read_image import read_image
from input_output.contours_io import write_contours
from gating.contour_based_gating import ContourBasedGating
from report.report import report
from segmentation.predict import Predict
from segmentation.segment import segment
from segmentation.save_as_nifti import save_as_nifti


class Master(QMainWindow):
    """Main Window Class

    Attributes:
        image: bool, indicates whether images have been loaded (true) or not
        contours: bool, indicates whether contours have been loaded (true) or not
        segmentation: bool, indicates whether segmentation has been performed (true) or not
        lumen: tuple, contours for lumen border
        plaque: tuple, contours for plaque border
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.autosave_interval = config.save.autosave_interval
        self.use_xml_files = config.save.use_xml_files
        self.contour_based_gating = ContourBasedGating(self)
        self.predictor = Predict(self)
        self.image_displayed = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.tmp_lumen_x = []  # for Ctrl+Z
        self.tmp_lumen_y = []
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}  # container to be saved in JSON file later, includes contours, etc.
        self.metadata = {}  # metadata used outside of readDICOM (not saved to JSON file)
        self.images = None
        self.diastole_color = (39, 69, 219)
        self.systole_color = (209, 55, 38)
        self.waiting_status = 'Waiting for user input...'
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self):
        SPACING = 5
        self.addToolBar('My Window')
        self.showMaximized()

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        main_window_hbox = QHBoxLayout()
        left_vbox = QVBoxLayout()
        right_vbox = QVBoxLayout()
        left_lower_hbox = QHBoxLayout()

        left_vbox.setContentsMargins(0, 0, SPACING, SPACING)
        right_vbox.setContentsMargins(SPACING, 0, 0, SPACING)
        right_upper_hbox = QHBoxLayout()
        right_lower_hbox = QHBoxLayout()
        right_vbox.addLayout(right_upper_hbox)
        right_vbox.addLayout(right_lower_hbox)
        main_window_hbox.addLayout(left_vbox)
        main_window_hbox.addLayout(right_vbox)

        image_button = QPushButton('Read DICOM/NIfTi')
        gating_button = QPushButton('Extract Diastolic and Systolic Frames')
        segment_button = QPushButton('Segment')
        contour_button = QPushButton('Manual Contour')
        write_button = QPushButton('Write Contours')
        report_button = QPushButton('Write Report')

        image_button.setToolTip('Load images in .dcm format')
        gating_button.setToolTip('Extract diastolic and systolic images from pullback')
        segment_button.setToolTip('Run deep learning based segmentation of lumen')
        contour_button.setToolTip('Manually draw new contour for lumen')
        write_button.setToolTip('Manually save contours in .json file')
        report_button.setToolTip('Write report to .txt file')

        vertical_header = QHeaderView(Qt.Vertical)
        vertical_header.hide()
        horizontal_header = QHeaderView(Qt.Horizontal)
        horizontal_header.hide()
        self.info_table = QTableWidget()
        self.info_table.setRowCount(8)
        self.info_table.setColumnCount(2)
        self.info_table.setItem(0, 0, QTableWidgetItem('Patient Name'))
        self.info_table.setItem(1, 0, QTableWidgetItem('Patient DOB'))
        self.info_table.setItem(2, 0, QTableWidgetItem('Patient Sex'))
        self.info_table.setItem(3, 0, QTableWidgetItem('Pullback Speed'))
        self.info_table.setItem(4, 0, QTableWidgetItem('Resolution (mm)'))
        self.info_table.setItem(5, 0, QTableWidgetItem('Dimensions'))
        self.info_table.setItem(6, 0, QTableWidgetItem('Manufacturer'))
        self.info_table.setItem(7, 0, QTableWidgetItem('Model'))
        self.info_table.setVerticalHeader(vertical_header)
        self.info_table.setHorizontalHeader(horizontal_header)
        self.info_table.horizontalHeader().setStretchLastSection(True)

        self.shortcut_info = QLabel()
        self.shortcut_info.setText(
            (
                '\n'
                'First, load a DICOM/NIfTi file using the button below or by pressing Ctrl+O.\n'
                'If available, contours for that file will be read automatically.\n'
                'Use the A and D keys to move through all frames, S and W keys to move through gated frames.\n'
                'Press E to draw a new Lumen contour.\n'
                'Press Delete to delete the current Lumen contour.\n'
                'Hold the right mouse button for windowing (can be reset by pressing R).\n'
                'Press C to toggle color mode.\n'
                'Press H to hide all contours.\n'
                'Press J to jiggle around the current frame.\n'
                'Press Ctrl+Q to close the program.\n'
            )
        )

        image_button.clicked.connect(lambda _: read_image(self))
        gating_button.clicked.connect(lambda _: self.contour_based_gating())
        segment_button.clicked.connect(lambda _: segment(self))
        contour_button.clicked.connect(lambda _: new_contour(self))
        write_button.clicked.connect(lambda _: write_contours(self))
        report_button.clicked.connect(lambda _: report(self))

        self.play_button = QPushButton()
        self.play_icon = self.style().standardIcon(getattr(QStyle, 'SP_MediaPlay'))
        self.pause_icon = self.style().standardIcon(getattr(QStyle, 'SP_MediaPause'))
        self.play_button.setIcon(self.play_icon)
        self.play_button.setMaximumWidth(30)
        self.play_button.clicked.connect(self.play)
        self.paused = True

        self.display_slider = Slider(Qt.Horizontal)
        self.display_slider.valueChanged[int].connect(self.change_value)

        max_box_width = 130
        self.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.diastolic_frame_box.setMaximumWidth(max_box_width)
        self.diastolic_frame_box.setChecked(False)
        self.diastolic_frame_box.stateChanged[int].connect(self.toggle_diastolic_frame)
        self.systolic_frame_box = QCheckBox('Systolic Frame')
        self.systolic_frame_box.setMaximumWidth(max_box_width)
        self.systolic_frame_box.setChecked(False)
        self.systolic_frame_box.stateChanged[int].connect(self.toggle_systolic_frame)
        self.plaque_frame_box = QCheckBox('Plaque')
        self.plaque_frame_box.setMaximumWidth(max_box_width)
        self.plaque_frame_box.setChecked(False)
        self.plaque_frame_box.stateChanged[int].connect(self.toggle_plaque_frame)

        self.hide_contours_box = QCheckBox('&Hide Contours')
        self.hide_contours_box.setChecked(False)
        self.hide_contours_box.stateChanged[int].connect(self.toggle_hide_contours)
        self.hide_special_points_box = QCheckBox('Hide farthest and closest points')
        self.hide_special_points_box.setChecked(False)
        self.hide_special_points_box.stateChanged[int].connect(self.toggle_hide_special_points)
        self.use_diastolic_button = QPushButton('Diastolic Frames')
        self.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.diastole_color}')
        self.use_diastolic_button.setCheckable(True)
        self.use_diastolic_button.setChecked(True)
        self.use_diastolic_button.clicked.connect(self.use_diastolic)
        self.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')

        self.display = Display(self, self.config)
        self.display_frame_comms = Communicate()
        self.display_frame_comms.updateBW[int].connect(self.display.set_frame)

        self.frame_number_label = QLabel()
        self.frame_number_label.setAlignment(Qt.AlignCenter)
        self.frame_number_label.setText(f'Frame {self.display_slider.value() + 1}')

        left_vbox.addWidget(self.display)
        left_lower_hbox.addWidget(self.play_button)
        left_lower_hbox.addWidget(self.display_slider)
        left_lower_hbox.addWidget(self.diastolic_frame_box)
        left_lower_hbox.addWidget(self.systolic_frame_box)
        left_lower_hbox.addWidget(self.plaque_frame_box)
        left_vbox.addLayout(left_lower_hbox)
        left_vbox.addWidget(self.frame_number_label)

        right_vbox.addWidget(self.hide_contours_box)
        right_vbox.addWidget(self.hide_special_points_box)
        right_vbox.addWidget(self.use_diastolic_button)
        right_vbox.addWidget(image_button)
        right_vbox.addWidget(gating_button)
        right_vbox.addWidget(segment_button)
        right_vbox.addWidget(contour_button)
        right_vbox.addWidget(write_button)
        right_vbox.addWidget(report_button)
        right_upper_hbox.addWidget(self.info_table)
        right_lower_hbox.addWidget(self.shortcut_info)

        central_widget = QWidget()
        central_widget.setLayout(main_window_hbox)
        self.setWindowIcon(QIcon('Media/thumbnail.png'))
        self.setWindowTitle('AAOCA Segmentation Tool')
        self.setCentralWidget(central_widget)
        self.show()

        timer = QTimer(self)
        timer.timeout.connect(self.auto_save)
        timer.start(self.autosave_interval)  # autosave interval in milliseconds

    def closeEvent(self, event):
        """Tasks to be performed before actually closing the program"""
        if self.image_displayed:
            self.save_before_close()

    def save_before_close(self):
        """Save contours, etc before closing program or reading new DICOM file"""
        self.status_bar.showMessage('Saving contours and NIfTi files...')
        write_contours(self)
        save_as_nifti(self)
        self.status_bar.showMessage(self.waiting_status)

    def play(self):
        """Plays all frames until end of pullback starting from currently selected frame"""
        start_frame = self.display_slider.value()

        if self.paused:
            self.paused = False
            self.play_button.setIcon(self.pause_icon)
        else:
            self.paused = True
            self.play_button.setIcon(self.play_icon)

        for frame in range(start_frame, self.metadata['num_frames']):
            if not self.paused:
                self.display_slider.setValue(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.frame_number_label.setText(f'Frame {frame + 1}')

        self.play_button.setIcon(self.play_icon)

    def auto_save(self):
        """Automatically saves contours to a temporary file every autoSaveInterval seconds"""
        if self.image_displayed:
            write_contours(self)

    def change_value(self, value):
        self.display_frame_comms.updateBW.emit(value)
        self.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')
        try:
            if self.data['plaque_frames'][value] == '1':
                self.plaque_frame_box.setChecked(True)
            else:
                self.plaque_frame_box.setChecked(False)
        except IndexError:
            pass

        try:
            if value in self.gated_frames_dia:
                self.diastolic_frame_box.setChecked(True)
            else:
                self.diastolic_frame_box.setChecked(False)
                if value in self.gated_frames_sys:
                    self.systolic_frame_box.setChecked(True)
                else:
                    self.systolic_frame_box.setChecked(False)
        except AttributeError:
            pass

    def toggle_hide_contours(self, value):
        if self.image_displayed:
            self.hide_contours = value
            self.display.update_display()

    def toggle_hide_special_points(self, value):
        if self.image_displayed:
            self.hide_special_points = value
            self.display.update_display()

    def use_diastolic(self):
        if self.image_displayed:
            if self.use_diastolic_button.isChecked():
                self.use_diastolic_button.setText('Diastolic Frames')
                self.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.diastole_color}')
                self.gated_frames = self.gated_frames_dia
            else:
                self.use_diastolic_button.setText('Systolic Frames')
                self.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.systole_color}')
                self.gated_frames = self.gated_frames_sys

            self.display_slider.set_gated_frames(self.gated_frames)

    def toggle_diastolic_frame(self, state_true):
        if self.image_displayed:
            frame = self.display_slider.value()
            if state_true:
                if frame not in self.gated_frames_dia:
                    bisect.insort_left(self.gated_frames_dia, frame)
                    self.data['phases'][frame] = 'D'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.systolic_frame_box.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.gated_frames_dia.remove(frame)
                    self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if self.use_diastolic_button.isChecked():
                self.display_slider.set_gated_frames(self.gated_frames_dia)

        self.display.display_image(update_phase=True)

    def toggle_systolic_frame(self, state_true):
        if self.image_displayed:
            frame = self.display_slider.value()
            if state_true:
                if frame not in self.gated_frames_sys:
                    bisect.insort_left(self.gated_frames_sys, frame)
                    self.data['phases'][frame] = 'S'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.diastolic_frame_box.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.gated_frames_sys.remove(frame)
                    self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if not self.use_diastolic_button.isChecked():
                self.display_slider.set_gated_frames(self.gated_frames_sys)

        self.display.display_image(update_phase=True)

    def toggle_plaque_frame(self, state_true):
        if self.image_displayed:
            frame = self.display_slider.value()
            if state_true:
                self.data['plaque_frames'][frame] = '1'
            else:
                self.data['plaque_frames'][frame] = '0'

    def errorMessage(self):
        """Helper function for errors"""

        warning = QMessageBox(self)
        warning.setWindowModality(Qt.WindowModal)
        warning.setWindowTitle('Error')
        warning.setText('Segmentation must be performed first')
        warning.exec_()

    def successMessage(self, task):
        """Helper function for success messages"""

        success = QMessageBox(self)
        success.setWindowModality(Qt.WindowModal)
        success.setWindowTitle('Status')
        success.setText(task + ' has been successfully completed')
        success.exec_()
