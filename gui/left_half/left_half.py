import time
import bisect

from loguru import logger
from functools import partial
from PyQt5.QtWidgets import QPushButton, QStyle, QApplication, QLabel, QCheckBox, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt

from gui.left_half.IVUS_display import IVUSDisplay
from gui.utils.slider import Slider, Communicate


class LeftHalf:
    def __init__(self, main_window):
        self.main_window = main_window

        main_window.display = IVUSDisplay(main_window)
        main_window.display_frame_comms = Communicate()
        main_window.display_frame_comms.updateBW[int].connect(main_window.display.set_frame)
        main_window.left_vbox.addWidget(main_window.display)

        left_lower_grid = QGridLayout()
        hide_checkboxes = QHBoxLayout()
        main_window.hide_contours_box = QCheckBox('&Hide Contours')
        main_window.hide_contours_box.setChecked(False)
        main_window.hide_contours_box.stateChanged[int].connect(self.toggle_hide_contours)
        main_window.hide_special_points_box = QCheckBox('Hide farthest and closest points')
        main_window.hide_special_points_box.setChecked(False)
        main_window.hide_special_points_box.stateChanged[int].connect(self.toggle_hide_special_points)
        hide_checkboxes.addWidget(main_window.hide_contours_box)
        hide_checkboxes.addWidget(main_window.hide_special_points_box)
        left_lower_grid.addLayout(hide_checkboxes, 0, 0)

        self.play_button = QPushButton()
        self.play_icon = main_window.style().standardIcon(getattr(QStyle, 'SP_MediaPlay'))
        self.pause_icon = main_window.style().standardIcon(getattr(QStyle, 'SP_MediaPause'))
        self.play_button.setIcon(self.play_icon)
        self.play_button.setMaximumWidth(30)
        self.play_button.clicked.connect(partial(self.play, main_window))
        self.paused = True
        main_window.display_slider = Slider(main_window, Qt.Horizontal)
        main_window.display_slider.valueChanged[int].connect(self.change_value)
        slider_hbox = QHBoxLayout()
        slider_hbox.addWidget(self.play_button)
        slider_hbox.addWidget(main_window.display_slider)
        left_lower_grid.addLayout(slider_hbox, 0, 1)

        self.frame_number_label = QLabel()
        self.frame_number_label.setAlignment(Qt.AlignCenter)
        self.frame_number_label.setText(f'Frame {main_window.display_slider.value() + 1}')
        frame_num_hbox = QHBoxLayout()
        frame_num_hbox.addWidget(self.frame_number_label)
        left_lower_grid.addLayout(frame_num_hbox, 1, 1)
        main_window.left_vbox.addLayout(left_lower_grid)

    def play(self, main_window):
        """Plays all frames until end of pullback starting from currently selected frame"""
        if not main_window.image_displayed:
            return

        start_frame = main_window.display_slider.value()
        if self.paused:
            self.paused = False
            self.play_button.setIcon(self.pause_icon)
        else:
            self.paused = True
            self.play_button.setIcon(self.play_icon)

        for frame in range(start_frame, main_window.metadata['num_frames']):
            if not self.paused:
                main_window.display_slider.set_value(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.frame_number_label.setText(f'Frame {frame + 1}')

        self.play_button.setIcon(self.play_icon)

    def change_value(self, value):
        self.main_window.display_frame_comms.updateBW.emit(value)
        self.main_window.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')

        if value in self.main_window.gated_frames_dia:
            self.main_window.diastolic_frame_box.setChecked(True)
        else:
            self.main_window.diastolic_frame_box.setChecked(False)
            if value in self.main_window.gated_frames_sys:
                self.main_window.systolic_frame_box.setChecked(True)
            else:
                self.main_window.systolic_frame_box.setChecked(False)

    def toggle_hide_contours(self, value):
        if self.main_window.image_displayed:
            self.main_window.hide_contours = value
            self.main_window.display.update_display()
            if not value:
                self.main_window.longitudinal_view.show_lview_contours()

    def toggle_hide_special_points(self, value):
        if self.main_window.image_displayed:
            self.main_window.hide_special_points = value
            self.main_window.display.update_display()
