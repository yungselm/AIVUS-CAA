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
        MAX_BOX_WIDTH = 130

        main_window.display = IVUSDisplay(main_window)
        main_window.display_frame_comms = Communicate()
        main_window.display_frame_comms.updateBW[int].connect(main_window.display.set_frame)
        main_window.left_vbox.addWidget(main_window.display)

        left_lower_grid = QGridLayout()
        flag_checkboxes = QHBoxLayout()
        self.diastolic_frame_box = QCheckBox('Diastolic Frame')
        self.diastolic_frame_box.setMaximumWidth(MAX_BOX_WIDTH)
        self.diastolic_frame_box.setChecked(False)
        self.diastolic_frame_box.stateChanged[int].connect(self.toggle_diastolic_frame)
        self.systolic_frame_box = QCheckBox('Systolic Frame')
        self.systolic_frame_box.setMaximumWidth(MAX_BOX_WIDTH)
        self.systolic_frame_box.setChecked(False)
        self.systolic_frame_box.stateChanged[int].connect(self.toggle_systolic_frame)
        self.plaque_frame_box = QCheckBox('Plaque')
        self.plaque_frame_box.setMaximumWidth(MAX_BOX_WIDTH)
        self.plaque_frame_box.setChecked(False)
        self.plaque_frame_box.stateChanged[int].connect(self.toggle_plaque_frame)
        flag_checkboxes.addWidget(self.diastolic_frame_box)
        flag_checkboxes.addWidget(self.systolic_frame_box)
        flag_checkboxes.addWidget(self.plaque_frame_box)
        left_lower_grid.addLayout(flag_checkboxes, 0, 0)
        hide_checkboxes = QHBoxLayout()
        main_window.hide_contours_box = QCheckBox('&Hide Contours')
        main_window.hide_contours_box.setChecked(False)
        main_window.hide_contours_box.stateChanged[int].connect(self.toggle_hide_contours)
        main_window.hide_special_points_box = QCheckBox('Hide farthest and closest points')
        main_window.hide_special_points_box.setChecked(False)
        main_window.hide_special_points_box.stateChanged[int].connect(self.toggle_hide_special_points)
        hide_checkboxes.addWidget(main_window.hide_contours_box)
        hide_checkboxes.addWidget(main_window.hide_special_points_box)
        left_lower_grid.addLayout(hide_checkboxes, 1, 0)

        self.play_button = QPushButton()
        self.play_icon = main_window.style().standardIcon(getattr(QStyle, 'SP_MediaPlay'))
        self.pause_icon = main_window.style().standardIcon(getattr(QStyle, 'SP_MediaPause'))
        self.play_button.setIcon(self.play_icon)
        self.play_button.setMaximumWidth(30)
        self.play_button.clicked.connect(partial(self.play, main_window))
        self.paused = True
        main_window.display_slider = Slider(Qt.Horizontal)
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
                main_window.display_slider.setValue(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.frame_number_label.setText(f'Frame {frame + 1}')

        self.play_button.setIcon(self.play_icon)

    def change_value(self, value):
        self.main_window.display_frame_comms.updateBW.emit(value)
        self.main_window.display.update_display()
        self.frame_number_label.setText(f'Frame {value + 1}')
        try:
            if self.main_window.data['plaque_frames'][value] == '1':
                self.plaque_frame_box.setChecked(True)
            else:
                self.plaque_frame_box.setChecked(False)
        except IndexError:
            pass

        try:
            if value in self.main_window.gated_frames_dia:
                self.diastolic_frame_box.setChecked(True)
            else:
                self.diastolic_frame_box.setChecked(False)
                if value in self.main_window.gated_frames_sys:
                    self.systolic_frame_box.setChecked(True)
                else:
                    self.systolic_frame_box.setChecked(False)
        except AttributeError:
            pass

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

    def toggle_diastolic_frame(self, state_true):
        if self.main_window.image_displayed:
            frame = self.main_window.display_slider.value()
            if state_true:
                if frame not in self.main_window.gated_frames_dia:
                    bisect.insort_left(self.main_window.gated_frames_dia, frame)
                    self.main_window.data['phases'][frame] = 'D'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.systolic_frame_box.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.main_window.gated_frames_dia.remove(frame)
                    if (
                        self.main_window.data['phases'][frame] == 'D'
                    ):  # do not reset when function is called from toggle_systolic_frame
                        self.main_window.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if self.main_window.use_diastolic_button.isChecked():
                self.main_window.display_slider.set_gated_frames(self.main_window.gated_frames_dia)

            self.main_window.display.update_display()

    def toggle_systolic_frame(self, state_true):
        if self.main_window.image_displayed:
            frame = self.main_window.display_slider.value()
            if state_true:
                if frame not in self.main_window.gated_frames_sys:
                    bisect.insort_left(self.main_window.gated_frames_sys, frame)
                    self.main_window.data['phases'][frame] = 'S'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.diastolic_frame_box.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.main_window.gated_frames_sys.remove(frame)
                    if (
                        self.main_window.data['phases'][frame] == 'S'
                    ):  # do not reset when function is called from toggle_diastolic_frame
                        self.main_window.data['phases'][frame] = '-'
                except ValueError:
                    pass
            if not self.main_window.use_diastolic_button.isChecked():
                self.main_window.display_slider.set_gated_frames(self.main_window.gated_frames_sys)

            self.main_window.display.update_display()

    def toggle_plaque_frame(self, state_true):
        if self.main_window.image_displayed:
            frame = self.main_window.display_slider.value()
            if state_true:
                self.main_window.data['plaque_frames'][frame] = '1'
            else:
                self.main_window.data['plaque_frames'][frame] = '0'
