import bisect

import numpy as np
from loguru import logger
from scipy.interpolate import splprep, splev
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QPainterPath, QColor


class Point(QGraphicsEllipseItem):
    """Class that describes a spline point"""

    def __init__(self, pos, line_thickness=1, point_radius=10, color=None, transparency=255):
        super(Point, self).__init__()
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.transparency = transparency
        self.default_color = get_qt_pen(color, line_thickness, transparency)

        self.setPen(self.default_color)
        self.setRect(
            pos[0] - self.point_radius * 0.5, pos[1] - self.point_radius * 0.5, self.point_radius, self.point_radius
        )

    def get_coords(self):
        try:
            return self.rect().x(), self.rect().y()
        except RuntimeError:  # Point has been deleted
            return None, None

    def update_color(self):
        self.setPen(QPen(Qt.transparent, self.line_thickness))

    def reset_color(self):
        self.setPen(self.default_color)

    def update_pos(self, pos):
        """Updates the Point position"""

        self.setRect(pos.x(), pos.y(), self.point_radius, self.point_radius)
        return self.rect()


class Spline(QGraphicsPathItem):
    """Class that describes a spline"""

    def __init__(self, points, n_points, line_thickness=1, color=None, transparency=255):
        super().__init__()
        self.n_points = n_points + 1
        self.knot_points = None
        self.full_contour = None
        self.set_knot_points(points)
        self.setPen(get_qt_pen(color, line_thickness, transparency))

    def set_knot_points(self, points):
        try:
            start_point = QPointF(points[0][0], points[1][0])
            self.path = QPainterPath(start_point)
            super(Spline, self).__init__(self.path)

            self.full_contour = self.interpolate(points)
            if self.full_contour[0] is not None:
                for i in range(0, len(self.full_contour[0])):
                    self.path.lineTo(self.full_contour[0][i], self.full_contour[1][i])

                self.setPath(self.path)
                self.path.closeSubpath()
                self.knot_points = points
        except IndexError:  # no points for this frame
            logger.error(points)
            pass

    def interpolate(self, points):
        """Interpolates the spline points at n_points points along spline"""
        points = np.array(points)
        try:
            tck, u = splprep(points, u=None, s=0.0, per=1)
        except ValueError:
            return (None, None)
        u_new = np.linspace(u.min(), u.max(), self.n_points)
        x_new, y_new = splev(u_new, tck, der=0)

        return (x_new, y_new)

    def update(self, pos, index, path_index=None):
        """Updates the stored spline everytime it is moved
        Args:
            pos: new points coordinates
            index: knot point index
            path_index: index of point on path
        """
        if path_index is not None:
            path_indices = np.zeros(len(self.knot_points[0]))
            distances = np.zeros(self.n_points)
            for i in range(len(self.knot_points[0])):
                knot_x, knot_y = self.knot_points[0][i], self.knot_points[1][i]
                for j in range(self.n_points):
                    distances[j] = np.sqrt(
                        (knot_x - self.full_contour[0][j]) ** 2 + (knot_y - self.full_contour[1][j]) ** 2
                    )
                path_indices[i] = np.argmin(distances)  # index of closest point on path
            path_indices[0] = 0  # first and last points are the same but need sorted list for bisect
            index = bisect.bisect_left(path_indices, path_index)
            self.knot_points[0].insert(index, pos.x())
            self.knot_points[1].insert(index, pos.y())
        else:
            if index >= len(self.knot_points[0]):
                self.knot_points[0].append(pos.x())
                self.knot_points[1].append(pos.y())
            else:
                self.knot_points[0][index] = pos.x()
                self.knot_points[1][index] = pos.y()
        self.full_contour = self.interpolate(self.knot_points)
        for i in range(0, len(self.full_contour[0])):
            self.path.setElementPositionAt(i, self.full_contour[0][i], self.full_contour[1][i])
        self.setPath(self.path)

        return index

    def on_path(self, pos):
        x, y = pos.x(), pos.y()
        distances = np.sqrt((self.full_contour[0] - x) ** 2 + (self.full_contour[1] - y) ** 2)
        if np.min(distances) < 10:
            return np.argmin(distances)
        return None

    def get_unscaled_contour(self, scaling_factor):
        return self.full_contour[0] / scaling_factor, self.full_contour[1] / scaling_factor

def get_qt_pen(color, thickness, transparency=255):
    try:
        color = getattr(Qt, color)
    except (AttributeError, TypeError):
        color = Qt.blue

    pen_color = QColor(color)
    pen_color.setAlpha(transparency)

    return QPen(pen_color, thickness)