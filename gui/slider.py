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

    def __init__(self, orientation):
        super().__init__()
        self.setOrientation(orientation)
        self.setRange(0, 0)
        self.setValue(0)
        self.setFocusPolicy(Qt.StrongFocus)
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Fixed)
        size_policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QSize(400, 25))
        self.setMaximumSize(QSize(1000, 25))
        self.gated_frames = []

    def next_frame(self):
        try:
            self.setValue(self.value() + 1)
        except IndexError:
            pass

    def last_frame(self):
        try:
            self.setValue(self.value() - 1)
        except IndexError:
            pass

    def next_gated_frame(self):
        if self.gated_frames:
            current_gated_frame = self.find_frame(self.value())
            if self.value() >= self.gated_frames[current_gated_frame]:
                current_gated_frame = current_gated_frame + 1
            try:
                self.setValue(self.gated_frames[current_gated_frame])
            except IndexError:
                pass
        else:
            self.next_frame()

    def last_gated_frame(self):
        if self.gated_frames:
            current_gated_frame = self.find_frame(self.value())
            if self.value() <= self.gated_frames[current_gated_frame]:
                current_gated_frame = current_gated_frame - 1
            if current_gated_frame < 0:
                current_gated_frame = 0
            self.setValue(self.gated_frames[current_gated_frame])
        else:
            self.last_frame()

    def find_frame(self, current_frame):
        """Find the closest gated frame"""
        gated_frames = np.asarray(self.gated_frames)
        closest_gated_frame = np.argmin(np.abs(gated_frames - current_frame))

        return closest_gated_frame

    def set_gated_frames(self, gated_frames):
        """Stores the gated frames."""
        self.gated_frames = gated_frames
