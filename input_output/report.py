import os
import math

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PyQt5.QtWidgets import QErrorMessage, QProgressDialog
from PyQt5.QtCore import Qt
from shapely.geometry import Polygon, Point, LineString
from itertools import combinations


def report(window):
    """Writes a report file containing lumen area, etc."""

    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot write report before reading DICOM file')
        warning.exec_()
        return

    if window.segmentation and not window.contours:
        window.errorMessage()
        return

    window.lumen = window.wid.getData()
    contoured_frames = [frame for frame in range(window.numberOfFrames) if window.lumen[0][frame]]

    longest_distances, shortest_distances, lumen_area = plotContoursWithMetrics(window, contoured_frames, plot=False)
    if longest_distances is None or shortest_distances is None:  # report was cancelled
        return

    f = open(os.path.splitext(window.file_name)[0] + "_report.txt", "w")
    f.write(
        "Frame\tPosition (mm)\tLumen area (mm\N{SUPERSCRIPT TWO})"
        "\tLongest Distance (mm)\t Longest x 1(px)\t Longest y 1(py)\t Longest x 2(px)"
        "\tShortest Distance (mm)\tElliptic Ratio\tPhase\n"
    )

    for index, frame in enumerate(contoured_frames):
        f.write(
            f"{frame}\t{window.pullbackLength[frame]:.2f}\t{lumen_area[index]:.2f}"
            f"\t{longest_distances[index]:.2f}\t{shortest_distances[index]:.2f}"
            f"\t{longest_distances[index]/shortest_distances[index]:.2f}\t{window.phases[frame]}\n"
        )
    f.close()

    window.successMessage("Write report")


def computeContourMetrics(window, lumen_x, lumen_y):
    """Computes lumen area, plaque area, and plaque burden from contours"""
    if lumen_x:
        lumen_area = contourArea(lumen_x, lumen_y) * window.resolution**2
        centroid_x, centroid_y = centroidPolygonSimple(lumen_x, lumen_y)

    return lumen_area, round(centroid_x), round(centroid_y)


def findShortestDistanceContour(window, polygon):
    centroid = polygon.centroid
    circle = Point(centroid).buffer(1)
    exterior_coords = polygon.exterior.coords[0::5]

    min_distance = math.inf
    closest_points = None
    min_dist_to_centroid = polygon.exterior.distance(centroid)

    for point1, point2 in combinations(exterior_coords, 2):
        distance = math.dist(point1, point2)
        if distance < min_distance and distance > min_dist_to_centroid * 2:
            line = LineString([point1, point2])
            if line.intersects(circle):
                min_distance = distance
                closest_points = (point1, point2)

    shortest_distance = min_distance * window.resolution

    # Separate x and y coordinates and append to the respective lists
    x1, y1 = closest_points[0]
    x2, y2 = closest_points[1]
    shortest_point_x = [x1, x2]
    shortest_point_y = [y1, y2]

    return shortest_distance, shortest_point_x, shortest_point_y


def findLongestDistanceContour(window, exterior_coords):
    max_distance = 0
    farthest_points = None

    for point1, point2 in combinations(exterior_coords, 2):
        distance = math.dist(point1, point2)
        if distance > max_distance:
            max_distance = distance
            farthest_points = (point1, point2)

    longest_distance = max_distance * window.resolution

    # Separate x and y coordinates and append to the respective lists
    x1, y1 = farthest_points[0]
    x2, y2 = farthest_points[1]
    longest_point_x = [x1, x2]
    longest_point_y = [y1, y2]

    return longest_distance, longest_point_x, longest_point_y


def contourArea(x, y):
    """Calculate contour/polygon area using Shoelace formula"""

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


# def contourEllipticRatio(x, y):
#     centroid

# def centroidPolygonComplex(area, x, y):


def centroidPolygonSimple(x, y):
    x = np.array(x)
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    return centroid_x, centroid_y


def plotContoursWithMetrics(window, contoured_frames, plot=True):
    """Plot contours and annotate with metrics"""
    progress = QProgressDialog()
    progress.setWindowFlags(Qt.Dialog)
    progress.setModal(True)
    progress.setMinimum(0)
    progress.setMaximum(len(contoured_frames) - 1)
    progress.resize(500, 100)
    progress.setValue(0)
    progress.setValue(1)
    progress.setValue(0)  # trick to make progress bar appear
    progress.setWindowTitle("Writing report...")
    progress.show()

    (  # initialise all needed lists
        longest_distances,
        longest_x,
        longest_y,
        shortest_distances,
        shortest_x,
        shortest_y,
        lumen_areas,
        centroids_x,
        centroids_y,
    ) = [[0] * len(contoured_frames) for _ in range(9)]
    for frame_index, frame in enumerate(contoured_frames):
        lumen_x, lumen_y = [window.lumen[i][frame] for i in range(2)]
        lumen_areas[frame_index], centroids_x[frame_index], centroids_y[frame_index] = computeContourMetrics(
            window, lumen_x, lumen_y
        )
        polygon = Polygon([(x, y) for x, y in zip(lumen_x, lumen_y)])
        exterior_coords = polygon.exterior.coords

        longest_distances[frame_index], longest_x[frame_index], longest_y[frame_index] = findLongestDistanceContour(
            window, exterior_coords
        )
        shortest_distances[frame_index], shortest_x[frame_index], shortest_y[frame_index] = findShortestDistanceContour(
            window, polygon
        )
        progress.setValue(frame_index)
        if progress.wasCanceled():
            break

    if progress.wasCanceled():
        return None, None

    progress.close()

    if plot:
        first_third = int(len(contoured_frames) * 0.25)
        second_third = int(len(contoured_frames) * 0.5)
        third_third = int(len(contoured_frames) * 0.75)
        indices_to_plot = [first_third, second_third, third_third]
        frames_to_plot = [contoured_frames[index] for index in indices_to_plot]

        for index, frame in zip(indices_to_plot, frames_to_plot):
            plt.figure(figsize=(6, 6))
            plt.plot(window.lumen[0][frame], window.lumen[1][frame], '-g', linewidth=2, label='Contour')
            plt.plot(centroids_x[index], centroids_y[index], 'ro', markersize=8, label='Centroid')
            plt.plot(longest_x[index][0], longest_y[index][0], 'bo', markersize=8, label='Farthest Point 1')
            plt.plot(longest_x[index][1], longest_y[index][1], 'bo', markersize=8, label='Farthest Point 2')
            plt.plot(shortest_x[index][0], shortest_y[index][0], 'yo', markersize=8, label='Shortest Point 1')
            plt.plot(shortest_x[index][1], shortest_y[index][1], 'yo', markersize=8, label='Shortest Point 2')

            # Annotate with shortest and longest distances
            plt.annotate(
                f'Shortest Distance: {shortest_distances[index]:.2f} mm',
                xy=(centroids_x[index], centroids_y[index]),
                xycoords='data',
                xytext=(10, 30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )

            plt.annotate(
                f'Longest Distance: {longest_distances[index]:.2f} mm',
                xy=(centroids_x[index], centroids_y[index]),
                xycoords='data',
                xytext=(10, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
            )

            plt.annotate(
                f'Lumen Area: {lumen_areas[index]:.2f} mm\N{SUPERSCRIPT TWO}\nElliptic Ratio: {longest_distances[index]/shortest_distances[index]:.2f}',
                xy=(centroids_x[index], centroids_y[index]),
                xycoords='data',
                xytext=(10, 0),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

            plt.title(f'Frame {frame}')
            plt.legend(loc='upper right')
            plt.gca().invert_yaxis()
            plt.grid()
            plt.tight_layout()
            plt.show()

    return longest_distances, shortest_distances, lumen_areas
