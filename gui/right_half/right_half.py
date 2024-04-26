import bisect

import matplotlib.pyplot as plt
from loguru import logger
from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSplitter, QPushButton, QCheckBox, QWidget

from gui.right_half.gating_display import GatingDisplay
from gui.right_half.longitudinal_view import LongitudinalView
from gui.utils.contours_gui import new_measure
from segmentation.segment import segment


class RightHalf:
    def __init__(self, main_window):
        self.main_window = main_window
        self.right_widget = QWidget()
        right_vbox = QVBoxLayout()
        checkboxes = QHBoxLayout()
        self.main_window.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.main_window.diastolic_frame_box.setChecked(False)
        self.main_window.diastolic_frame_box.stateChanged[int].connect(partial(toggle_diastolic_frame, main_window))
        self.main_window.systolic_frame_box = QCheckBox('Systolic Frame')
        self.main_window.systolic_frame_box.setChecked(False)
        self.main_window.systolic_frame_box.stateChanged[int].connect(partial(toggle_systolic_frame, main_window))
        checkboxes.addWidget(self.main_window.diastolic_frame_box)
        checkboxes.addWidget(self.main_window.systolic_frame_box)
        main_window.gating_display = GatingDisplay(main_window)
        checkboxes.addWidget(main_window.gating_display.toolbar)
        right_vbox.addLayout(checkboxes)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(main_window.gating_display)
        main_window.longitudinal_view = LongitudinalView(main_window)
        splitter.addWidget(main_window.longitudinal_view)
        gating_display_size = main_window.gating_display.sizeHint().height()
        splitter.setSizes([gating_display_size, gating_display_size])
        splitter.setStretchFactor(0, main_window.config.display.gating_display_stretch)
        splitter.setStretchFactor(1, main_window.config.display.lview_display_stretch)
        right_vbox.addWidget(splitter)

        right_lower_vbox = QVBoxLayout()
        self.main_window.use_diastolic_button = QPushButton('Diastolic Frames')
        self.main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{self.main_window.diastole_color}')
        self.main_window.use_diastolic_button.setCheckable(True)
        self.main_window.use_diastolic_button.setChecked(True)
        self.main_window.use_diastolic_button.clicked.connect(partial(use_diastolic, main_window))
        self.main_window.use_diastolic_button.setToolTip('Press button to switch between diastolic and systolic frames')
        segment_button = QPushButton('Automatic Segmentation')
        segment_button.setToolTip('Run deep learning based segmentation of lumen')
        segment_button.clicked.connect(partial(segment, main_window))
        gating_button = QPushButton('Extract Diastolic and Systolic Frames')
        gating_button.setToolTip('Extract diastolic and systolic images from pullback')
        gating_button.clicked.connect(main_window.contour_based_gating)
        measure_button_1 = QPushButton('Measurement &1')
        measure_button_1.setToolTip('Measure distance between two points')
        measure_button_1.clicked.connect(partial(new_measure, main_window, index=0))
        measure_button_1.setStyleSheet(f'border-color: {main_window.measure_colors[0]}')
        measure_button_2 = QPushButton('Measurement &2')
        measure_button_2.setToolTip('Measure distance between two points')
        measure_button_2.clicked.connect(partial(new_measure, main_window, index=1))
        measure_button_2.setStyleSheet(f'border-color: {main_window.measure_colors[1]}')
        right_lower_vbox.addWidget(self.main_window.use_diastolic_button)
        command_buttons = QHBoxLayout()
        command_buttons.addWidget(segment_button)
        command_buttons.addWidget(gating_button)
        right_lower_vbox.addLayout(command_buttons)
        measures = QHBoxLayout()
        measures.addWidget(measure_button_1)
        measures.addWidget(measure_button_2)
        right_lower_vbox.addLayout(measures)
        right_vbox.addLayout(right_lower_vbox)
        self.right_widget.setLayout(right_vbox)

    def __call__(self):
        return self.right_widget


def toggle_diastolic_frame(main_window, state_true):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            if frame not in main_window.gated_frames_dia:
                bisect.insort_left(main_window.gated_frames_dia, frame)
                main_window.data['phases'][frame] = 'D'
                main_window.contour_based_gating.update_color(main_window.diastole_color_plt)
                plt.draw()
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.systolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.gated_frames_dia.remove(frame)
                if (
                    main_window.data['phases'][frame] == 'D'
                ):  # do not reset when function is called from toggle_systolic_frame
                    main_window.data['phases'][frame] = '-'
                    main_window.contour_based_gating.update_color()
            except ValueError:
                pass
        if main_window.use_diastolic_button.isChecked():
            main_window.display_slider.set_gated_frames(main_window.gated_frames_dia)

        main_window.display.update_display()


def toggle_systolic_frame(main_window, state_true):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            if frame not in main_window.gated_frames_sys:
                bisect.insort_left(main_window.gated_frames_sys, frame)
                main_window.data['phases'][frame] = 'S'
                main_window.contour_based_gating.update_color(main_window.systole_color_plt)
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.diastolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.gated_frames_sys.remove(frame)
                if (
                    main_window.data['phases'][frame] == 'S'
                ):  # do not reset when function is called from toggle_diastolic_frame
                    main_window.data['phases'][frame] = '-'
                    main_window.contour_based_gating.update_color()
            except ValueError:
                pass
        if not main_window.use_diastolic_button.isChecked():
            main_window.display_slider.set_gated_frames(main_window.gated_frames_sys)

        main_window.display.update_display()


def use_diastolic(main_window):
    if main_window.image_displayed:
        if main_window.use_diastolic_button.isChecked():
            main_window.use_diastolic_button.setText('Diastolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.diastole_color}')
            main_window.gated_frames = main_window.gated_frames_dia
        else:
            main_window.use_diastolic_button.setText('Systolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.systole_color}')
            main_window.gated_frames = main_window.gated_frames_sys

        main_window.display_slider.set_gated_frames(main_window.gated_frames)
