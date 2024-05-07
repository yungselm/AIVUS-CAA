from loguru import logger
from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPen
from shapely.geometry import Polygon

from gui.utils.geometry import Spline, Point
from report.report import farthest_points, closest_points


class SmallDisplay(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_size = main_window.config.display.image_size
        self.n_points_contour = main_window.config.display.n_points_contour
        self.contour_thickness = main_window.config.display.contour_thickness
        self.point_thickness = main_window.config.display.point_thickness
        self.point_radius = main_window.config.display.point_radius
        self.scaling_factor = self.image_size / self.main_window.images[0].shape[0]
        self.window_to_image_ratio = 1.5
        self.window_size = int(self.image_size / self.window_to_image_ratio)
        self.resize(self.window_size, self.window_size)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
        )

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCentralWidget(self.view)
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def set_frame(self, frame):
        contour_types = (Spline, Point, QGraphicsLineItem)  # types of items to remove from scene
        [self.scene.removeItem(item) for item in self.scene.items() if isinstance(item, contour_types)]

        if frame is None:
            self.pixmap.setPixmap(QPixmap())
            self.setWindowTitle("No Frame to Display")
            return

        self.pixmap.setPixmap(
            QPixmap.fromImage(
                QImage(
                    self.main_window.images[frame],
                    self.main_window.images[frame].shape[1],
                    self.main_window.images[frame].shape[0],
                    self.main_window.images[frame].shape[1],
                    QImage.Format_Grayscale8,
                ).scaled(self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            )
        )
        if self.main_window.data['lumen'][0][frame]:
            lumen_x = [point * self.scaling_factor for point in self.main_window.data['lumen'][0][frame]]
            lumen_y = [point * self.scaling_factor for point in self.main_window.data['lumen'][1][frame]]
            current_contour = Spline([lumen_x, lumen_y], self.n_points_contour, self.contour_thickness, 'green')

            if current_contour.full_contour[0] is not None:
                self.contour_points = [
                    Point(
                        (current_contour.knot_points[0][i], current_contour.knot_points[1][i]),
                        self.point_thickness,
                        self.point_radius,
                        'green',
                    )
                    for i in range(len(current_contour.knot_points[0]) - 1)
                ]
                [self.scene.addItem(point) for point in self.contour_points]
                self.scene.addItem(current_contour)
                polygon = Polygon(
                    [(x, y) for x, y in zip(current_contour.full_contour[0], current_contour.full_contour[1])]
                )
                self.view.centerOn(polygon.centroid.x, polygon.centroid.y)
                _, farthest_x, farthest_y = farthest_points(self.main_window, polygon.exterior.coords, frame)
                _, closest_x, closest_y = closest_points(self.main_window, polygon, frame)
                self.scene.addLine(
                    farthest_x[0],
                    farthest_y[0],
                    farthest_x[1],
                    farthest_y[1],
                    QPen(Qt.yellow, self.point_thickness * 2),
                )
                self.scene.addLine(
                    closest_x[0],
                    closest_y[0],
                    closest_x[1],
                    closest_y[1],
                    QPen(Qt.yellow, self.point_thickness * 2),
                )

        current_phase = 'Diastolic' if self.main_window.use_diastolic_button.isChecked() else 'Systolic'
        self.setWindowTitle(f"Next {current_phase} Frame {frame + 1}")
