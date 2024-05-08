import numpy as np
from loguru import logger
from PyQt5.QtWidgets import (
    QSlider,
    QSizePolicy,
)
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QSize


class Communicate(QObject):
    updateBW = pyqtSignal(int)
    updateBool = pyqtSignal(bool)


class Slider(QSlider):
    """Slider for changing the currently displayed image."""

    def __init__(self, main_window, orientation):
        super().__init__()
        self.main_window = main_window
        self.setOrientation(orientation)
        self.setRange(0, 0)
        self.setFocusPolicy(Qt.StrongFocus)
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Fixed)
        size_policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QSize(200, 25))
        self.setMaximumSize(QSize(1000, 25))

    def set_value(self, value, reset_highlights=True):
        self.setValue(value)
        try:  # small display
            next_gated = self.next_gated_frame(set=False)
            self.main_window.small_display.set_frame(next_gated)
        except AttributeError:
            pass
        try:  # gating display
            if reset_highlights:
                self.main_window.contour_based_gating.reset_highlights()
            self.main_window.contour_based_gating.set_frame(value)
        except AttributeError:
            pass


    def next_frame(self):
        try:
            self.set_value(self.value() + 1)
        except IndexError:
            pass

    def last_frame(self):
        try:
            self.set_value(self.value() - 1)
        except IndexError:
            pass

    def next_gated_frame(self, set=True):
        if self.main_window.gated_frames:
            current_gated_frame = self.find_frame(self.value())
            if self.value() >= self.main_window.gated_frames[current_gated_frame]:
                current_gated_frame = current_gated_frame + 1
            try:
                if set:
                    self.set_value(self.main_window.gated_frames[current_gated_frame])
                else:
                    return self.main_window.gated_frames[current_gated_frame]
            except IndexError:
                return None
        else:
            if set:
                self.next_frame()
            else:
                return None

    def last_gated_frame(self, set=True):
        if self.main_window.gated_frames:
            current_gated_frame = self.find_frame(self.value())
            if self.value() <= self.main_window.gated_frames[current_gated_frame]:
                current_gated_frame = current_gated_frame - 1
            if current_gated_frame < 0:
                current_gated_frame = 0
            if set:
                self.set_value(self.main_window.gated_frames[current_gated_frame])
            else:
                return self.main_window.gated_frames[current_gated_frame]
        else:
            if set:
                self.last_frame()
            else:
                return None

    def find_frame(self, current_frame):
        """Find the closest gated frame"""
        gated_frames = np.asarray(self.main_window.gated_frames)
        closest_gated_frame = np.argmin(np.abs(gated_frames - current_frame))

        return closest_gated_frame
