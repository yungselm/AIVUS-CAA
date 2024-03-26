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
        self.lview_contour_size = 2
        self.graphics_scene = QGraphicsScene()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setScene(self.graphics_scene)

    def set_data(self, images, contours):
        self.graphics_scene.clear()
        self.num_frames = images.shape[0]
        self.points_on_marker = [None] * self.num_frames
        self.image_width = images.shape[1]

        slice = images[:, self.image_width // 2, :]
        slice = np.flipud(np.transpose(slice, (1, 0))).copy()  # need .copy() to avoid QImage TypeError
        longitudinal_image = QImage(
            slice.data, self.num_frames, self.image_width, self.num_frames, QImage.Format_Grayscale8
        )
        image = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(image)

        for frame, contour in enumerate(contours):
            self.lview_contour(frame, contour)

    def update_marker(self, frame):
        [self.graphics_scene.removeItem(item) for item in self.graphics_scene.items() if isinstance(item, Marker)]
        marker = Marker(frame, 0, frame, self.image_width)
        self.graphics_scene.addItem(marker)

    def lview_contour(self, frame, contour, scaling_factor=1, update=False):
        index = None
        if self.points_on_marker[frame] is not None:  # remove previous points
            for point in self.points_on_marker[frame]:
                self.graphics_scene.removeItem(point)

        if contour is None:  # skip frames without contour (but still remove previous points)
            return

        if update or self.points_on_marker[frame] is None:  # need to find the two closest points to the marker
            distances = contour.full_contour[0] / scaling_factor - self.image_width // 2
            num_points_to_collect = len(contour.full_contour[0]) // 10
            point_indices = np.argpartition(np.abs(distances), num_points_to_collect)[:num_points_to_collect]
            for i in range(len(point_indices)):
                if (
                    np.abs(contour.full_contour[1][point_indices[0]] - contour.full_contour[1][point_indices[i]])
                    > self.image_width / 10
                ):  # ensure the two points are from different sides of the contour
                    index = i
                    break
            if index is None:  # no suitable points found
                return
            self.points_on_marker[frame] = (
                Point(
                    (frame, contour.full_contour[1][point_indices[0]] / scaling_factor),
                    line_thickness=self.lview_contour_size,
                    point_radius=self.lview_contour_size,
                    color='g',
                ),
                Point(
                    (frame, contour.full_contour[1][point_indices[index]] / scaling_factor),
                    line_thickness=self.lview_contour_size,
                    point_radius=self.lview_contour_size,
                    color='g',
                ),
            )
        for point in self.points_on_marker[frame]:
            self.graphics_scene.addItem(point)


class Marker(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color=Qt.white):
        super().__init__()
        pen = QPen(QColor(color), 1)
        pen.setDashPattern([1, 6])
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
