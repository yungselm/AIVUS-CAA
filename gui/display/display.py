import math

import numpy as np
from loguru import logger
import cv2
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from shapely.geometry import Polygon

from gui.geometry import Point, Spline
from report.report import compute_polygon_metrics, farthest_points, closest_points
from segmentation.segment import downsample


class Display(QGraphicsView):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window, config):
        super(Display, self).__init__()
        self.main_window = main_window
        self.n_interactive_points = config.display.n_interactive_points
        self.n_points_contour = config.display.n_points_contour
        self.image_size = config.display.image_size
        self.windowing_sensitivity = config.display.windowing_sensitivity
        self.contour_thickness = config.display.contour_thickness
        self.point_thickness = config.display.point_thickness
        self.point_radius = config.display.point_radius
        self.graphics_scene = QGraphicsScene(self)
        self.point_index = None
        self.frame = 0
        self.lumen_contour = None  # entire contour (not only knotpoints), needed for elliptic ratio
        self.draw = False
        self.points_to_draw = []
        self.contour_drawn = False
        self.new_spline = None
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
        self.frame_metrics_text = None
        self.graphics_scene.addItem(self.image)
        self.setScene(self.graphics_scene)

    def set_data(self, lumen, images):
        self.num_frames = images.shape[0]
        self.scaling_factor = self.image_size / images.shape[1]
        if (
            lumen[0] and max([len(lumen[0][frame]) for frame in range(self.num_frames)]) > self.n_interactive_points
        ):  # contours with higher number of knotpoints loaded -> save downsampled version
            self.main_window.data['lumen'] = downsample(lumen, self.n_interactive_points)
        else:
            self.main_window.data['lumen'] = lumen
        self.images = images
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def display_image(self, update_image=False, update_contours=False, update_phase=False):
        """Clears scene and displays current image and contours"""
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
            normalised_data = np.clip(
                self.images[self.frame, :, :], lower_bound, upper_bound
            )  # clip values to be within the range
            normalised_data = ((normalised_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            height, width = normalised_data.shape

            if self.main_window.colormap_enabled:
                # Apply an orange-blue colormap
                colormap = cv2.applyColorMap(normalised_data, cv2.COLORMAP_JET)
                q_image = QImage(colormap.data, width, height, width * 3, QImage.Format.Format_RGB888).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
            else:
                q_image = QImage(normalised_data.data, width, height, width, QImage.Format.Format_Grayscale8).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )

            self.q_image = q_image  # Update the QImage
            self.pixmap = QPixmap.fromImage(q_image)
            self.image = QGraphicsPixmapItem(self.pixmap)
            self.graphics_scene.addItem(self.image)

        old_contours = [
            item for item in self.graphics_scene.items() if isinstance(item, (Spline, Point, QGraphicsTextItem))
        ]
        [
            self.graphics_scene.removeItem(item)
            for item in self.graphics_scene.items()
            if isinstance(item, (Spline, Point, QGraphicsTextItem))
        ]  # clear previous scene
        if not self.main_window.hide_contours:
            if update_contours:
                self.draw_contour(self.main_window.data['lumen'])
                if self.main_window.data['lumen'][0][self.frame] and self.lumen_contour.full_contour[0] is not None:
                    lumen_x = [point / self.scaling_factor for point in self.lumen_contour.full_contour[0]]
                    lumen_y = [point / self.scaling_factor for point in self.lumen_contour.full_contour[1]]
                    polygon = Polygon([(x, y) for x, y in zip(lumen_x, lumen_y)])
                    lumen_area, lumen_circumf, _, _ = compute_polygon_metrics(self.main_window, polygon, self.frame)
                    longest_distance, farthest_point_x, farthest_point_y = farthest_points(
                        self.main_window, polygon.exterior.coords, self.frame
                    )
                    shortest_distance, closest_point_x, closest_point_y = closest_points(
                        self.main_window, polygon, self.frame
                    )
                    if not self.main_window.hide_special_points:
                        for i in range(2):  # draw farthest and closest points
                            self.graphics_scene.addItem(
                                Point(
                                    (
                                        farthest_point_x[i] * self.scaling_factor,
                                        farthest_point_y[i] * self.scaling_factor,
                                    ),
                                    self.point_thickness * 2,
                                    self.point_radius,
                                    'r',
                                )
                            )
                            self.graphics_scene.addItem(
                                Point(
                                    (
                                        closest_point_x[i] * self.scaling_factor,
                                        closest_point_y[i] * self.scaling_factor,
                                    ),
                                    self.point_thickness * 2,
                                    self.point_radius,
                                    'y',
                                )
                            )

                    elliptic_ratio = (longest_distance / shortest_distance) if shortest_distance != 0 else 0
                    self.frame_metrics_text = QGraphicsTextItem(
                        f'Lumen area:\t{round(lumen_area, 2)} (mm\N{SUPERSCRIPT TWO})\n'
                        f'Lumen circumf:\t{round(lumen_circumf, 2)} (mm)\n'
                        f'Elliptic ratio:\t{round(elliptic_ratio, 2)}'
                    )
                    self.frame_metrics_text.setFont(QFont('Helvetica', int(self.image_size / 50)))
                    self.graphics_scene.addItem(self.frame_metrics_text)
                    if not update_phase:
                        self.graphics_scene.addItem(self.phase_text)
            else:  # re-draw old elements to put them in foreground
                [self.graphics_scene.addItem(item) for item in old_contours]

        if update_phase:
            if self.main_window.data['phases'][self.frame] == 'D':
                phase = 'Diastole'
                color = QColor(
                    self.main_window.diastole_color[0],
                    self.main_window.diastole_color[1],
                    self.main_window.diastole_color[2],
                )
            elif self.main_window.data['phases'][self.frame] == 'S':
                phase = 'Systole'
                color = QColor(
                    self.main_window.systole_color[0],
                    self.main_window.systole_color[1],
                    self.main_window.systole_color[2],
                )
            else:
                phase = ''
                color = Qt.white
            self.phase_text = QGraphicsTextItem(phase)
            self.phase_text.setDefaultTextColor(color)
            self.phase_text.setPos(0, self.image_size / 8.5)
            self.phase_text.setFont(QFont('Helvetica', int(self.image_size / 50), QFont.Bold))
            self.graphics_scene.addItem(self.phase_text)

    def draw_contour(self, lumen):
        """Adds lumen contours to scene"""

        if lumen[0][self.frame]:
            lumen_x = [point * self.scaling_factor for point in lumen[0][self.frame]]
            lumen_y = [point * self.scaling_factor for point in lumen[1][self.frame]]
            self.lumen_contour = Spline([lumen_x, lumen_y], self.n_points_contour, self.contour_thickness, 'g')
            if self.lumen_contour.full_contour[0] is not None:
                self.lumen_points = [
                    Point(
                        (self.lumen_contour.knot_points[0][idx], self.lumen_contour.knot_points[1][idx]),
                        self.point_thickness,
                        self.point_radius,
                        'g',
                    )
                    for idx in range(len(self.lumen_contour.knot_points[0]) - 1)
                ]
                [self.graphics_scene.addItem(point) for point in self.lumen_points]
                self.graphics_scene.addItem(self.lumen_contour)
            else:
                logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated')

    def add_manual_contour(self, point):
        """Creates an interactive contour manually point by point"""

        if self.points_to_draw:
            start_point = self.points_to_draw[0].get_coords()
        else:
            self.contour_drawn = False
            start_point = (point.x(), point.y())

        if start_point[0] is None:  # occurs when Point has been deleted during draw (e.g. by RMB click)
            self.points_to_draw = []
            self.draw = False
            self.main_window.setCursor(Qt.ArrowCursor)
            self.display_image(update_contours=True)
        else:
            self.points_to_draw.append(Point((point.x(), point.y()), self.point_thickness, self.point_radius))
            self.graphics_scene.addItem(self.points_to_draw[-1])

            if len(self.points_to_draw) > 3:
                if not self.contour_drawn:
                    self.new_spline = Spline(
                        [
                            [point.get_coords()[0] for point in self.points_to_draw],
                            [point.get_coords()[1] for point in self.points_to_draw],
                        ],
                        self.n_points_contour,
                        self.contour_thickness,
                    )
                    self.graphics_scene.addItem(self.new_spline)
                    self.contour_drawn = True
                else:
                    self.new_spline.update(point, len(self.points_to_draw))

            if len(self.points_to_draw) > 1:
                dist = math.sqrt(
                    (point.x() - self.points_to_draw[0].get_coords()[0]) ** 2
                    + (point.y() - self.points_to_draw[0].get_coords()[1]) ** 2
                )

                if dist < 20:
                    self.draw = False
                    self.points_to_draw = []
                    if self.new_spline is not None:
                        downsampled = downsample(
                            ([self.new_spline.full_contour[0].tolist()], [self.new_spline.full_contour[1].tolist()]),
                            self.n_interactive_points,
                        )
                        self.main_window.data['lumen'][0][self.frame] = [
                            point / self.scaling_factor for point in downsampled[0][0]
                        ]
                        self.main_window.data['lumen'][1][self.frame] = [
                            point / self.scaling_factor for point in downsampled[1][0]
                        ]

                    self.main_window.setCursor(Qt.ArrowCursor)
                    self.display_image(update_contours=True)

    def update_display(self):
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def start_drawing(self):
        self.main_window.setCursor(Qt.CrossCursor)
        self.draw = True
        self.points_to_draw = []
        self.main_window.data['lumen'][0][self.frame] = []
        self.main_window.data['lumen'][1][self.frame] = []

        self.display_image(update_contours=True)

    def set_frame(self, value):
        self.frame = value

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.draw:
                pos = self.mapToScene(event.pos())
                self.add_manual_contour(pos)
            else:
                # identify which point has been clicked
                items = self.items(event.pos())
                for item in items:
                    if item in self.lumen_points:
                        self.main_window.setCursor(Qt.BlankCursor)  # remove cursor for precise contour changes
                        # Convert mouse position to item position
                        # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
                        self.point_index = self.lumen_points.index(item)
                        item.update_color()
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
                new_point = item.update_pos(mouse_position)
                self.lumen_contour.update(new_point, self.point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # for some reason event.buttons() does not work here
            if self.point_index is not None:
                self.main_window.setCursor(Qt.ArrowCursor)
                item = self.active_point
                item.reset_color()

                self.main_window.data['lumen'][0][self.frame] = [
                    point / self.scaling_factor for point in self.lumen_contour.knot_points[0]
                ]
                self.main_window.data['lumen'][1][self.frame] = [
                    point / self.scaling_factor for point in self.lumen_contour.knot_points[1]
                ]
                self.display_image(update_contours=True)
                self.point_index = None
