import time

import numpy as np
from loguru import logger
from PyQt5.QtWidgets import (
    QSlider,
    QApplication,
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

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_D:
            self.setValue(self.value() + 1)
        elif key == Qt.Key_Left or key == Qt.Key_A:
            self.setValue(self.value() - 1)
        elif key == Qt.Key_Up or key == Qt.Key_W:
            self.next_gated_frame()
        elif key == Qt.Key_Down or key == Qt.Key_S:
            self.last_gated_frame()
        elif key == Qt.Key_J:
            self.setValue(self.value() - 1)
            QApplication.processEvents()
            time.sleep(0.1)
            self.setValue(self.value() + 1)
            QApplication.processEvents()
            time.sleep(0.1)
            self.setValue(self.value() + 1)
            QApplication.processEvents()
            time.sleep(0.1)
            self.setValue(self.value() - 1)
            QApplication.processEvents()

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
            try:
                self.setValue(self.value() + 1)
            except IndexError:
                pass
    def last_gated_frame(self):
        if self.gated_frames:
            current_gated_frame = self.find_frame(self.value())
            if self.value() <= self.gated_frames[current_gated_frame]:
                current_gated_frame = current_gated_frame - 1
            if current_gated_frame < 0:
                current_gated_frame = 0
            self.setValue(self.gated_frames[current_gated_frame])
        else:
            self.setValue(self.value() - 1)

    def find_frame(self, current_frame):
        """Find the closest gated frame"""
        gated_frames = np.asarray(self.gated_frames)
        closest_gated_frame = np.argmin(np.abs(gated_frames - current_frame))

        return closest_gated_frame

    def set_gated_frames(self, gated_frames):
        """Stores the gated frames."""
        self.gated_frames = gated_frames