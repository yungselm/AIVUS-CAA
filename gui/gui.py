import time
import bisect

from loguru import logger
from functools import partial
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
    QMenuBar,
    QStatusBar,
    QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from gui.display.IVUS_display import IVUSDisplay
from gui.display.longitudinal_view import LongitudinalView
from gui.slider import Slider, Communicate
from gui.shortcuts import (
    init_shortcuts,
    open_url,
    hide_contours,
    hide_special_points,
    toggle_filter,
    reset_windowing,
    toggle_color,
)
from gui.display.contours_gui import new_contour, new_measure
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
        self.filter = None
        self.tmp_lumen_x = []  # for Ctrl+Z
        self.tmp_lumen_y = []
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}  # container to be saved in JSON file later, includes contours, etc.
        self.metadata = {}  # metadata used outside of read_image (not saved to JSON file)
        self.images = None
        self.diastole_color = (39, 69, 219)
        self.systole_color = (209, 55, 38)
        self.measure_colors = ['red', 'cyan']
        self.waiting_status = 'Waiting for user input...'
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self):
        SPACING = 5
        self.showMaximized()

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        file_menu = self.menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open File', partial(read_image, self))
        open_action.setShortcut('Ctrl+O')
        file_menu.addSeparator()
        file_menu.addAction('Save Contours', partial(write_contours, self))
        file_menu.addAction('Save Report', partial(report, self))
        file_menu.addSeparator()
        exit_action = file_menu.addAction('Exit', self.close)
        exit_action.setShortcut('Ctrl+Q')

        edit_menu = self.menu_bar.addMenu('Edit')
        manual_contour = edit_menu.addAction('Manual Contour', partial(new_contour, self))
        manual_contour.setShortcut('E')
        edit_menu.addSeparator()
        measure_1 = edit_menu.addAction('Measurement 1', partial(new_measure, self, index=0))
        measure_1.setShortcut('1')
        measure_2 = edit_menu.addAction('Measurement 2', partial(new_measure, self, index=1))
        measure_2.setShortcut('2')

        view_menu = self.menu_bar.addMenu('View')
        hide_contours_action = view_menu.addAction('Hide Contours', partial(hide_contours, self))
        hide_contours_action.setShortcut('H')
        hide_special_points_action = view_menu.addAction('Hide Special Points', partial(hide_special_points, self))
        hide_special_points_action.setShortcut('G')
        view_menu.addSeparator()
        reset_windowing_action = view_menu.addAction('Reset Windowing', partial(reset_windowing, self))
        reset_windowing_action.setShortcut('R')
        toggle_color_action = view_menu.addAction('Toggle Color', partial(toggle_color, self))
        toggle_color_action.setShortcut('C')
        view_menu.addSeparator()
        filter_1 = view_menu.addAction('Apply Median Blur', partial(toggle_filter, self, index=0))
        filter_1.setShortcut('3')
        filter_2 = view_menu.addAction('Apply Gaussian Blur', partial(toggle_filter, self, index=1))
        filter_2.setShortcut('4')
        filter_3 = view_menu.addAction('Apply Bilateral Filter', partial(toggle_filter, self, index=2))
        filter_3.setShortcut('5')

        run_menu = self.menu_bar.addMenu('Run')
        run_menu.addAction('Extract Diastolic and Systolic Frames', self.contour_based_gating)
        run_menu.addAction('Automatic Segmentation', partial(segment, self))

        help_menu = self.menu_bar.addMenu('Help')
        help_menu.addAction('GitHub Page', partial(open_url, self, description='github'))
        help_menu.addAction('Keyboard Shortcuts', partial(open_url, self, description='keyboard_shortcuts'))
        help_menu.addSeparator()
        help_menu.addAction('About', partial(open_url, self))

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        main_window_hbox = QHBoxLayout()
        left_vbox = QVBoxLayout()
        right_vbox = QVBoxLayout()

        left_vbox.setContentsMargins(0, 0, SPACING, SPACING)
        right_vbox.setContentsMargins(SPACING, 0, 0, SPACING)
        # right_upper_hbox = QHBoxLayout()
        right_middle_hbox = QHBoxLayout()
        right_lower_vbox = QVBoxLayout()
        # right_vbox.addLayout(right_upper_hbox, stretch=1)
        right_vbox.addLayout(right_middle_hbox, stretch=2)
        right_vbox.addLayout(right_lower_vbox)
        main_window_hbox.addLayout(left_vbox)
        main_window_hbox.addLayout(right_vbox)

        self.display = IVUSDisplay(self, self.config)
        self.display_frame_comms = Communicate()
        self.display_frame_comms.updateBW[int].connect(self.display.set_frame)

        # vertical_header = QHeaderView(Qt.Vertical)
        # vertical_header.hide()
        # horizontal_header = QHeaderView(Qt.Horizontal)
        # horizontal_header.hide()
        self.info_table = QTableWidget()
        # self.info_table.setRowCount(8)
        # self.info_table.setColumnCount(2)
        # self.info_table.setItem(0, 0, QTableWidgetItem('Patient Name'))
        # self.info_table.setItem(1, 0, QTableWidgetItem('Patient DOB'))
        # self.info_table.setItem(2, 0, QTableWidgetItem('Patient Sex'))
        # self.info_table.setItem(3, 0, QTableWidgetItem('Pullback Speed'))
        # self.info_table.setItem(4, 0, QTableWidgetItem('Resolution (mm)'))
        # self.info_table.setItem(5, 0, QTableWidgetItem('Dimensions'))
        # self.info_table.setItem(6, 0, QTableWidgetItem('Manufacturer'))
        # self.info_table.setItem(7, 0, QTableWidgetItem('Model'))
        # self.info_table.setVerticalHeader(vertical_header)
        # self.info_table.setHorizontalHeader(horizontal_header)
        # self.info_table.horizontalHeader().setStretchLastSection(True)

        gating_button = QPushButton('Extract Diastolic and Systolic Frames')
        gating_button.setToolTip('Extract diastolic and systolic images from pullback')
        gating_button.clicked.connect(self.contour_based_gating)
        segment_button = QPushButton('Automatic Segmentation')
        segment_button.setToolTip('Run deep learning based segmentation of lumen')
        segment_button.clicked.connect(partial(segment, self))
        measure_button_1 = QPushButton('Measurement &1')
        measure_button_1.setToolTip('Measure distance between two points')
        measure_button_1.clicked.connect(partial(new_measure, self, index=0))
        measure_button_1.setStyleSheet(f'border-color: {self.measure_colors[0]}')
        measure_button_2 = QPushButton('Measurement &2')
        measure_button_2.setToolTip('Measure distance between two points')
        measure_button_2.clicked.connect(partial(new_measure, self, index=1))
        measure_button_2.setStyleSheet(f'border-color: {self.measure_colors[1]}')

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

        self.longitudinal_view = LongitudinalView(self, self.config)

        self.frame_number_label = QLabel()
        self.frame_number_label.setAlignment(Qt.AlignCenter)
        self.frame_number_label.setText(f'Frame {self.display_slider.value() + 1}')

        left_vbox.addWidget(self.display)
        left_lower_grid = QGridLayout()
        flag_checkboxes = QHBoxLayout()
        flag_checkboxes.addWidget(self.diastolic_frame_box)
        flag_checkboxes.addWidget(self.systolic_frame_box)
        flag_checkboxes.addWidget(self.plaque_frame_box)
        left_lower_grid.addLayout(flag_checkboxes, 0, 0)
        slider_hbox = QHBoxLayout()
        slider_hbox.addWidget(self.play_button)
        slider_hbox.addWidget(self.display_slider)
        left_lower_grid.addLayout(slider_hbox, 0, 1)
        hide_checkboxes = QHBoxLayout()
        hide_checkboxes.addWidget(self.hide_contours_box)
        hide_checkboxes.addWidget(self.hide_special_points_box)
        left_lower_grid.addLayout(hide_checkboxes, 1, 0)
        frame_num_hbox = QHBoxLayout()
        frame_num_hbox.addWidget(self.frame_number_label)
        left_lower_grid.addLayout(frame_num_hbox, 1, 1)
        left_vbox.addLayout(left_lower_grid)

        # right_upper_hbox.addWidget(self.info_table)
        right_middle_hbox.addWidget(self.longitudinal_view)
        right_lower_vbox.addWidget(self.use_diastolic_button)
        command_buttons = QHBoxLayout()
        right_lower_vbox.addLayout(command_buttons)
        command_buttons.addWidget(gating_button)
        command_buttons.addWidget(segment_button)
        measures = QHBoxLayout()
        right_lower_vbox.addLayout(measures)
        measures.addWidget(measure_button_1)
        measures.addWidget(measure_button_2)

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
        if not self.image_displayed:
            return

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
                    if (
                        self.data['phases'][frame] == 'D'
                    ):  # do not reset when function is called from toggle_systolic_frame
                        self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if self.use_diastolic_button.isChecked():
                self.display_slider.set_gated_frames(self.gated_frames_dia)

        self.display.update_display()

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
                    if (
                        self.data['phases'][frame] == 'S'
                    ):  # do not reset when function is called from toggle_diastolic_frame
                        self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if not self.use_diastolic_button.isChecked():
                self.display_slider.set_gated_frames(self.gated_frames_sys)

        self.display.update_display()

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
