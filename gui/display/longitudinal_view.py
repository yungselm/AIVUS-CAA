import numpy as np
from loguru import logger
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen


class LongitudinalView(QGraphicsView):
    """
    Displays the longitudinal view of the IVUS pullback.
    """

    def __init__(self, main_window, config):
        super().__init__()
        self.main_window = main_window
        self.image_size = config.display.image_size
        self.graphics_scene = QGraphicsScene()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setScene(self.graphics_scene)

    def set_data(self, images, frame):
        self.graphics_scene.clear()
        self.num_frames = images.shape[0]
        self.image_width = images.shape[1]

        slice = images[:, self.image_width // 2, :]
        slice = np.transpose(slice, (1, 0)).copy()  # need .copy() to avoid QImage TypeError
        longitudinal_image = QImage(
            slice.data, self.num_frames, self.image_width, self.num_frames, QImage.Format_Grayscale8
        )
        image = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(image)

        marker = Marker(frame, 0, frame, self.image_width)
        self.graphics_scene.addItem(marker)


class Marker(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color=Qt.white):
        super().__init__()
        pen = QPen(QColor(color), 2)
        pen.setDashPattern([1, 6])
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
