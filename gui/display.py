import math

import numpy as np
from loguru import logger
import cv2
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from shapely.geometry import Polygon

from gui.geometry import Point, Spline
from input_output.report import computeContourMetrics, findLongestDistanceContour, findShortestDistanceContour


class Display(QGraphicsView):
    """Displays images and contours.

    Displays images and contours as well as allowing user to
    interact and manipulate contours.

    Attributes:
        graphics_scene: QGraphicsScene, all items
        frame: int, current frame
        lumen: tuple, lumen contours
        hide_contours: bool, indicates whether contours should be displayed or hidden
        activePoint: Point, active point in spline
        innerPoint: list, spline points for inner (lumen) contours
    """

    def __init__(self, main_window, config):
        super(Display, self).__init__()
        self.main_window = main_window
        self.image_size = config.display.image_size
        self.windowing_sensitivity = config.display.windowing_sensitivity
        self.spline_thickness = config.display.spline_thickness
        self.point_thickness = config.display.point_thickness
        self.point_radius = config.display.point_radius
        self.graphics_scene = QGraphicsScene(self)
        self.point_index = None
        self.frame = 0
        self.hide_contours = True
        self.lumen_spline = None  # entire contour (not only knotpoints), needed for elliptic ratio
        self.draw = False
        self.drawPoints = []
        self.splineDrawn = False
        self.newSpline = None
        self.enable_drag = True
        self.active_point = None
        self.lumen_points = []

        # Store initial window level and window width (full width, middle level)
        self.initial_window_level = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width = 256  # window width is the range of pixel values that are displayed
        self.window_level = self.initial_window_level
        self.window_width = self.initial_window_width

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.image = QGraphicsPixmapItem(QPixmap(self.image_size, self.image_size))
        self.text = None
        self.graphics_scene.addItem(self.image)
        self.setScene(self.graphics_scene)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.draw:
                pos = self.mapToScene(event.pos())
                self.addManualSpline(pos)
            else:
                # identify which point has been clicked
                items = self.items(event.pos())
                for item in items:
                    if item in self.lumen_points:
                        self.main_window.setCursor(Qt.BlankCursor)  # remove cursor for precise spline changes
                        # Convert mouse position to item position
                        # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
                        self.point_index = self.lumen_points.index(item)
                        item.updateColor()
                        self.enable_drag = True
                        self.active_point = item

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.mouse_x = event.x()
            self.mouse_y = event.y()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.point_index is not None:
                item = self.active_point
                mouse_position = item.mapFromScene(self.mapToScene(event.pos()))
                new_point = item.update(mouse_position)
                self.lumen_spline.update(new_point, self.point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.y() - self.mouse_y) * self.windowing_sensitivity
            self.displayImage(update_image=True)
            self.setMouseTracking(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # for some reason event.buttons() does not work here
            if self.point_index is not None:
                self.main_window.setCursor(Qt.ArrowCursor)
                item = self.active_point
                item.resetColor()

                self.main_window.data['lumen'][0][self.frame] = [
                    val / self.scaling_factor for val in self.lumen_spline.knotpoints[0]
                ]
                self.main_window.data['lumen'][1][self.frame] = [
                    val / self.scaling_factor for val in self.lumen_spline.knotpoints[1]
                ]
                self.displayImage(update_splines=True)
                self.point_index = None

    def setData(self, lumen, images):
        self.numberOfFrames = images.shape[0]
        self.scaling_factor = self.image_size / images.shape[1]
        if len(lumen[0][0]) == 500:  # complete contour loaded -> save downsampled version
            self.main_window.data['lumen'] = self.downsample(lumen)
        else:
            self.main_window.data['lumen'] = lumen
        self.images = images
        self.displayImage(update_image=True, update_splines=True)

    def downsample(self, contours, num_points=20):
        """Downsamples input contour data by selecting n points from original contour"""

        num_frames = len(contours[0])
        downsampled = [[] for _ in range(num_frames)], [[] for _ in range(num_frames)]

        for frame in range(num_frames):
            if contours[0][frame]:
                points_to_sample = range(0, len(contours[0][frame]), len(contours[0][frame]) // num_points)
                for axis in range(2):
                    downsampled[axis][frame] = [contours[axis][frame][point] for point in points_to_sample]
        return downsampled

    def displayImage(self, update_image=False, update_splines=False):
        """Clears scene and displays current image and splines"""
        if update_image:
            [
                self.graphics_scene.removeItem(item)
                for item in self.graphics_scene.items()
                if isinstance(item, QGraphicsPixmapItem)
            ]  # clear previous scene
            self.active_point = None
            self.point_index = None

            # Calculate lower and upper bounds for the adjusted window level and window width
            lower_bound = self.window_level - self.window_width / 2
            upper_bound = self.window_level + self.window_width / 2

            # Clip and normalize pixel values
            normalized_data = np.clip(
                self.images[self.frame, :, :], lower_bound, upper_bound
            )  # clip values to be within the range
            normalized_data = ((normalized_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            height, width = normalized_data.shape

            if self.main_window.colormap_enabled:
                # Apply an orange-blue colormap
                colormap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
                q_image = QImage(colormap.data, width, height, width * 3, QImage.Format.Format_RGB888).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
            else:
                q_image = QImage(normalized_data.data, width, height, width, QImage.Format.Format_Grayscale8).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )

            self.q_image = q_image  # Update the QImage
            self.pixmap = QPixmap.fromImage(q_image)
            self.image = QGraphicsPixmapItem(self.pixmap)
            self.graphics_scene.addItem(self.image)

        old_splines = [
            item for item in self.graphics_scene.items() if isinstance(item, (Spline, Point, QGraphicsTextItem))
        ]
        [
            self.graphics_scene.removeItem(item)
            for item in self.graphics_scene.items()
            if isinstance(item, (Spline, Point, QGraphicsTextItem))
        ]  # clear previous scene
        if not self.hide_contours:
            if update_splines:
                self.addInteractiveSplines(self.main_window.data['lumen'])
                lumen_x, lumen_y = [list(self.lumen_spline.full_contour[i]) for i in range(2)]
                polygon = Polygon([(x, y) for x, y in zip(lumen_x, lumen_y)])
                lumen_area, _, _ = computeContourMetrics(self.main_window, lumen_x, lumen_y, self.frame)

                longest_distance, _, _ = findLongestDistanceContour(
                    self.main_window, polygon.exterior.coords, self.frame
                )
                shortest_distance, _, _ = findShortestDistanceContour(self.main_window, polygon, self.frame)

                elliptic_ratio = (longest_distance / shortest_distance) if shortest_distance != 0 else 0
                self.text = QGraphicsTextItem(
                    f"Lumen area:\t{round(lumen_area, 2)} (mm\N{SUPERSCRIPT TWO})\n"
                    f"Lumen circumf:\t{round(polygon.length * self.main_window.metadata['resolution'], 2)} (mm)\n"
                    f"Elliptic ratio:\t{round(elliptic_ratio, 2)}"
                )
                self.graphics_scene.addItem(self.text)
            else:  # re-draw old elements to put them in foreground
                [self.graphics_scene.addItem(item) for item in old_splines]

    def addInteractiveSplines(self, lumen):
        """Adds lumen splines to scene"""

        if lumen[0][self.frame]:
            lumen_x = [val * self.scaling_factor for val in lumen[0][self.frame]]
            lumen_y = [val * self.scaling_factor for val in lumen[1][self.frame]]
            self.lumen_spline = Spline([lumen_x, lumen_y], 'g', self.spline_thickness)
            self.lumen_points = [
                Point(
                    (self.lumen_spline.knotpoints[0][idx], self.lumen_spline.knotpoints[1][idx]),
                    'g',
                    self.point_thickness,
                    self.point_radius,
                )
                for idx in range(len(self.lumen_spline.knotpoints[0]) - 1)
            ]
            [self.graphics_scene.addItem(point) for point in self.lumen_points]
            self.graphics_scene.addItem(self.lumen_spline)

    def addManualSpline(self, point):
        """Creates an interactive spline manually point by point"""

        if self.drawPoints:
            start_point = self.drawPoints[0].getPoint()
        else:
            self.splineDrawn = False
            start_point = (point.x(), point.y())

        if start_point[0] is None:  # occurs when Point has been deleted during draw (e.g. by RMB click)
            self.drawPoints = []
            self.draw = False
            self.main_window.setCursor(Qt.ArrowCursor)
            self.displayImage(update_splines=True)
        else:
            self.drawPoints.append(Point((point.x(), point.y()), 'b', self.point_thickness, self.point_radius))
            self.graphics_scene.addItem(self.drawPoints[-1])

            if len(self.drawPoints) > 3:
                if not self.splineDrawn:
                    self.newSpline = Spline(
                        [
                            [point.getPoint()[0] for point in self.drawPoints],
                            [point.getPoint()[1] for point in self.drawPoints],
                        ],
                        'c',
                        self.spline_thickness,
                    )
                    self.graphics_scene.addItem(self.newSpline)
                    self.splineDrawn = True
                else:
                    self.newSpline.update(point, len(self.drawPoints))

            if len(self.drawPoints) > 1:
                dist = math.sqrt(
                    (point.x() - self.drawPoints[0].getPoint()[0]) ** 2
                    + (point.y() - self.drawPoints[0].getPoint()[1]) ** 2
                )

                if dist < 20:
                    self.draw = False
                    self.drawPoints = []
                    if self.newSpline is not None:
                        downsampled = self.downsample(
                            ([self.newSpline.full_contour[0].tolist()], [self.newSpline.full_contour[1].tolist()])
                        )
                        self.main_window.data['lumen'][0][self.frame] = [
                            val / self.scaling_factor for val in downsampled[0][0]
                        ]
                        self.main_window.data['lumen'][1][self.frame] = [
                            val / self.scaling_factor for val in downsampled[1][0]
                        ]

                    self.main_window.setCursor(Qt.ArrowCursor)
                    self.displayImage(update_splines=True)

    def run(self):
        self.displayImage(update_image=True, update_splines=True)

    def new_contour(self, main_window):
        self.main_window = main_window
        self.main_window.setCursor(Qt.CrossCursor)
        self.draw = True
        self.drawPoints = []

        self.main_window.data['lumen'][0][self.frame] = []
        self.main_window.data['lumen'][1][self.frame] = []

        self.displayImage(update_splines=True)

    def setFrame(self, value):
        self.frame = value

    def setDisplay(self, hide_contours):
        self.hide_contours = hide_contours
