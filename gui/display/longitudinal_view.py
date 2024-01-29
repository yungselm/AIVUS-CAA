import numpy as np
from loguru import logger
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen

from gui.geometry import Point


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

    def set_data(self, images):
        self.graphics_scene.clear()
        self.num_frames = images.shape[0]
        self.image_width = images.shape[1]

        slice = images[:, self.image_width // 2, :]
        slice = np.flipud(np.transpose(slice, (1, 0))).copy()  # need .copy() to avoid QImage TypeError
        longitudinal_image = QImage(
            slice.data, self.num_frames, self.image_width, self.num_frames, QImage.Format_Grayscale8
        )
        image = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(image)

    def update_marker(self, frame):
        [self.graphics_scene.removeItem(item) for item in self.graphics_scene.items() if isinstance(item, Marker)]
        marker = Marker(frame, 0, frame, self.image_width)
        self.graphics_scene.addItem(marker)

    def update_contour(self, x_pos, contours):
        [self.graphics_scene.removeItem(item) for item in self.graphics_scene.items() if isinstance(item, Point)]
        for frame, contour in enumerate(contours):
            distances = contour.full_contour[0] - x_pos
            points_on_marker = np.argpartition(np.abs(distances), 2)[:2]  # get the two closest points to the marker
            for point in points_on_marker:
                point = Point((frame, contour.full_contour[1][point]), line_thickness=2, point_radius=2, color='g')
                self.graphics_scene.addItem(point)


class Marker(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color=Qt.white):
        super().__init__()
        pen = QPen(QColor(color), 2)
        pen.setDashPattern([1, 6])
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
