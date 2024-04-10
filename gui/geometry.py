import numpy as np
from loguru import logger
from scipy.interpolate import splprep, splev
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QPainterPath


class Point(QGraphicsEllipseItem):
    """Class that describes a spline point"""

    def __init__(self, pos, line_thickness=1, point_radius=10, color=None):
        super(Point, self).__init__()
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.default_color = get_qt_pen(color, line_thickness)

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

        self.setRect(
            pos.x(), pos.y(), self.point_radius, self.point_radius
        )
        return self.rect()


class Spline(QGraphicsPathItem):
    """Class that describes a spline"""

    def __init__(self, points, n_points, line_thickness=1, color=None):
        super().__init__()
        self.n_points = n_points + 1
        self.knot_points = None
        self.full_contour = None
        self.set_knot_points(points)
        self.setPen(get_qt_pen(color, line_thickness))

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

    def update(self, pos, idx):
        """Updates the stored spline everytime it is moved
        Args:
            pos: new points coordinates
            idx: index on spline of updated point
        """

        if idx == len(self.knot_points[0]) + 1:
            self.knot_points[0].append(pos.x())
            self.knot_points[1].append(pos.y())
        else:
            self.knot_points[0][idx] = pos.x()
            self.knot_points[1][idx] = pos.y()
        self.full_contour = self.interpolate(self.knot_points)
        for i in range(0, len(self.full_contour[0])):
            self.path.setElementPositionAt(i, self.full_contour[0][i], self.full_contour[1][i])
        self.setPath(self.path)


def get_qt_pen(color, thickness):
    if color == 'yellow':
        return QPen(Qt.yellow, thickness)
    elif color == 'red':
        return QPen(Qt.red, thickness)
    elif color == 'green':
        return QPen(Qt.green, thickness)
    elif color == 'cyan':
        return QPen(Qt.cyan, thickness)
    else:
        return QPen(Qt.blue, thickness)
