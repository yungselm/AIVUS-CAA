import math

import numpy as np
from loguru import logger
import cv2
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QImage

from gui.geometry import Point, Spline


class Display(QGraphicsView):
    """Displays images and contours.

    Displays images and contours as well as allowing user to
    interact and manipulate contours.

    Attributes:
        scene: QGraphicsScene, all items
        frame: int, current frame
        lumen: tuple, lumen contours
        hide: bool, indicates whether contours should be displayed or hidden
        activePoint: Point, active point in spline
        innerPoint: list, spline points for inner (lumen) contours
    """

    def __init__(self, window, windowing_sensitivity):
        super(Display, self).__init__()
        print("View Height: {}, View Width: {}".format(self.width(), self.height()))

        scene = QGraphicsScene(self)
        self.window = window
        self.windowing_sensitivity = windowing_sensitivity
        self.scene = scene
        self.pointIdx = None
        self.frame = 0
        self.lumen = ([], [])
        self.hide = True
        self.draw = False
        self.drawPoints = []
        self.splineDrawn = False
        self.newSpline = None
        self.enable_drag = True
        self.activePoint = None
        self.innerPoint = []
        self.display_size = 800

        # Store initial window level and window width (full width, middle level)
        self.initial_window_level = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width = 256  # window width is the range of pixel values that are displayed
        self.window_level = self.initial_window_level
        self.window_width = self.initial_window_width

        self.viewport().installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.image = QGraphicsPixmapItem(QPixmap(self.display_size, self.display_size))
        self.scene.addItem(self.image)
        self.setScene(self.scene)

    def eventFilter(self, obj, event):
        """Handle mouse events for adjusting window level and window width"""
        if event.type() == QEvent.Type.MouseMove and event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            dx = (event.x() - self.mouse_x) * self.windowing_sensitivity
            dy = (event.y() - self.mouse_y) * self.windowing_sensitivity

            self.window_level += dx
            self.window_width += dy

            self.displayImage()

        elif event.type() == QEvent.Type.MouseButtonPress and event.buttons() == Qt.MouseButton.RightButton:
            self.mouse_x = event.x()
            self.mouse_y = event.y()
        
        self.setMouseTracking(False)

        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        super(Display, self).mousePressEvent(event)

        if self.draw:
            pos = self.mapToScene(event.pos())
            self.addManualSpline(pos)
        else:
            # identify which point has been clicked
            items = self.items(event.pos())
            for item in items:
                if item in self.innerPoint:
                    # Convert mouse position to item position https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
                    self.pointIdx = [i for i, checkItem in enumerate(self.innerPoint) if item == checkItem][0]
                    item.updateColor()
                    self.enable_drag = True
                    self.activePoint = item

    def mouseReleaseEvent(self, event):
        if self.pointIdx is not None:
            contour_scaling_factor = self.display_size / self.imsize[1]
            item = self.activePoint
            item.resetColor()

            self.lumen[0][self.frame] = [val / contour_scaling_factor for val in self.innerSpline.knotPoints[0]]
            self.lumen[1][self.frame] = [val / contour_scaling_factor for val in self.innerSpline.knotPoints[1]]

    def mouseMoveEvent(self, event):
        if self.pointIdx is not None:
            item = self.activePoint
            pos = item.mapFromScene(self.mapToScene(event.pos()))
            newPos = item.update(pos)
            # update the spline
            self.innerSpline.update(newPos, self.pointIdx)

    def setData(self, lumen, images):
        self.numberOfFrames = images.shape[0]
        self.lumen = self.downsample(lumen)
        self.images = images
        self.imsize = self.images.shape
        self.displayImage()

    def getData(self):
        """Gets the interpolated image contours

        Returns:
            lumenContour: list, first and second lists are lists of x and y points
        """

        lumenContour = [[], []]

        for frame in range(self.numberOfFrames):
            if self.lumen[0][frame]:
                lumen = Spline([self.lumen[0][frame], self.lumen[1][frame]], 'g')
                lumenContour[0].append(list(lumen.points[0]))
                lumenContour[1].append(list(lumen.points[1]))
            else:
                lumenContour[0].append([])
                lumenContour[1].append([])

        return lumenContour

    def downsample(self, contours, num_points=20):
        """Downsamples input contour data by selecting n points from original contour"""

        numberOfFrames = len(contours[0])
        downsampled = [[] for _ in range(numberOfFrames)], [[] for _ in range(numberOfFrames)]

        for i in range(numberOfFrames):
            if contours[0][i]:
                idx = len(contours[0][i]) // num_points
                downsampled[0][i] = [pnt for j, pnt in enumerate(contours[0][i]) if j % idx == 0]
                downsampled[1][i] = [pnt for j, pnt in enumerate(contours[1][i]) if j % idx == 0]

        return downsampled

    def displayImage(self):
        """Clears scene and displays current image and splines"""

        self.scene.clear()
        self.viewport().update()

        [self.removeItem(item) for item in self.scene.items()]

        self.activePoint = None
        self.pointIdx = None

        # Calculate lower and upper bounds for the adjusted window level and window width
        lower_bound = self.window_level - self.window_width / 2
        upper_bound = self.window_level + self.window_width / 2

        # Clip and normalize pixel values
        normalized_data = np.clip(
            self.images[self.frame, :, :], lower_bound, upper_bound
        )  # clip values to be within the range
        normalized_data = ((normalized_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
        height, width = normalized_data.shape

        if self.window.colormap_enabled:
            # Apply an orange-blue colormap
            colormap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
            q_image = QImage(colormap.data, width, height, width * 3, QImage.Format.Format_RGB888).scaled(
                self.display_size, self.display_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
        else:
            q_image = QImage(normalized_data.data, width, height, width, QImage.Format.Format_Grayscale8).scaled(
                self.display_size, self.display_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )

        self.q_image = q_image  # Update the QImage
        self.pixmap = QPixmap.fromImage(q_image)
        self.image = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image)

        if not self.hide:
            if self.lumen[0]:
                self.addInteractiveSplines(self.lumen)

        self.setScene(self.scene)

    def addInteractiveSplines(self, lumen):
        """Adds inner and outer splines to scene"""

        contour_scaling_factor = self.display_size / self.imsize[1]
        if lumen[0][self.frame]:
            lumen_x = [val * contour_scaling_factor for val in lumen[0][self.frame]]
            lumen_y = [val * contour_scaling_factor for val in lumen[1][self.frame]]
            self.innerSpline = Spline([lumen_x, lumen_y], 'g')
            self.innerPoint = [
                Point((self.innerSpline.knotPoints[0][idx], self.innerSpline.knotPoints[1][idx]), 'g')
                for idx in range(len(self.innerSpline.knotPoints[0]) - 1)
            ]
            [self.scene.addItem(point) for point in self.innerPoint]
            self.scene.addItem(self.innerSpline)

    def addManualSpline(self, point):
        """Creates an interactive spline manually point by point"""

        if not self.drawPoints:
            self.splineDrawn = False

        self.drawPoints.append(Point((point.x(), point.y()), 'b'))
        self.scene.addItem(self.drawPoints[-1])

        if len(self.drawPoints) > 3:
            if not self.splineDrawn:
                self.newSpline = Spline(
                    [
                        [point.getPoint()[0] for point in self.drawPoints],
                        [point.getPoint()[1] for point in self.drawPoints],
                    ],
                    'c',
                )
                self.scene.addItem(self.newSpline)
                self.splineDrawn = True
            else:
                self.newSpline.update(point, len(self.drawPoints))

        if len(self.drawPoints) > 1:
            dist = math.sqrt(
                (point.x() - self.drawPoints[0].getPoint()[0]) ** 2
                + (point.y() - self.drawPoints[0].getPoint()[1]) ** 2
            )

            if dist < 10:
                self.draw = False
                self.drawPoints = []
                if self.newSpline is not None:
                    downsampled = self.downsample(
                        ([self.newSpline.points[0].tolist()], [self.newSpline.points[1].tolist()])
                    )
                    scaling_factor = self.display_size / self.imsize[1]
                    self.lumen[0][self.frame] = [val / scaling_factor for val in downsampled[0][0]]
                    self.lumen[1][self.frame] = [val / scaling_factor for val in downsampled[1][0]]

                self.win.setCursor(Qt.ArrowCursor)
                self.displayImage()

    def run(self):
        self.displayImage()

    def new(self, window):
        self.win = window
        self.win.setCursor(Qt.CrossCursor)

        self.draw = True

        self.lumen[0][self.frame] = []
        self.lumen[1][self.frame] = []

        self.displayImage()

    def setFrame(self, value):
        self.frame = value

    def setDisplay(self, hide):
        self.hide = hide
