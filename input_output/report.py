import os
import math
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PyQt5.QtWidgets import QErrorMessage
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
    contoured_frames = [
        frame for frame in range(window.numberOfFrames) if window.lumen[0][frame]
    ]
    lumen_area, centroid_x, centroid_y = computeContourMetrics(window, contoured_frames)
    
    longest_distances, longest_x, longest_y = findLongestDistanceContour(window, contoured_frames)
    shortest_distances, shortest_x, shortest_y = findShortestDistanceContour(window, contoured_frames)


    print(longest_x[0][0])
    print(longest_y[0][0])
    print(shortest_distances[0])

    plotContoursWithMetrics(window, contoured_frames, centroid_x, centroid_y, lumen_area, longest_distances, shortest_distances)

    f = open(os.path.splitext(window.file_name)[0] + "_report.txt", "w")
    f.write(
        "Frame\tPosition (mm)\tLumen area (mm\N{SUPERSCRIPT TWO})"
        "\tCentroid x (px)\tCentroid y (px)\tLongest Distance (mm)\t Longest x 1(px)\t Longest y 1(py)\t Longest x 2(px)"
        "\t Longest y 2(py)\tShortest Distance (mm)\t Shortest x (px)\t Shortest y\tPhase\n"
    )

    for index, frame in enumerate(contoured_frames):
        f.write(
            "{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\n".format(
                frame,
                window.pullbackLength[frame],
                lumen_area[index],
                centroid_x[index],
                centroid_y[index],
                longest_distances[index],
                longest_x[index][0],
                longest_y[index][0],
                longest_x[index][1],
                longest_y[index][1],
                shortest_distances[index],
                shortest_x[index][0],
                shortest_y[index][0],
                window.phases[frame],
            )
        )
    f.close()

    window.successMessage("Write report")


def computeContourMetrics(window, contoured_frames):
    """Computes lumen area, plaque area, and plaque burden from contours"""

    lumen_area = np.zeros(len(contoured_frames))
    centroid_0x = np.zeros(len(contoured_frames))
    centroid_0y = np.zeros(len(contoured_frames))
    
    for index, frame in enumerate(contoured_frames):
        if window.lumen[0][frame]:
            lumen_area[index] = contourArea(window.lumen[0][frame], window.lumen[1][frame]) * window.resolution**2
            centroid_x, centroid_y = centroidPolygonSimple(window.lumen[0][frame], window.lumen[1][frame])
            centroid_0x[index] = round(centroid_x)
            centroid_0y[index] = round(centroid_y)

    return (lumen_area, centroid_0x, centroid_0y)


def findShortestDistanceContour(window, contoured_frames):
    shortest_distances = []
    shortest_points_x = []
    shortest_points_y = []

    for frame in contoured_frames:
        polygon = Polygon([(x, y) for x, y in zip(window.lumen[0][frame], window.lumen[1][frame])])
        centroid = polygon.centroid
        circle = Point(centroid).buffer(1)
        exterior_coords = polygon.exterior.coords[0::3]
        print(len(exterior_coords))

        min_distance = math.inf
        nearest_points = None

        for point1, point2 in combinations(exterior_coords, 2):
            line = LineString([point1, point2])
            if line.intersects(circle):
                distance = math.dist(point1, point2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_points = (point1, point2)

        shortest_distances.append(min_distance * window.resolution)
        
        # Separate x and y coordinates and append to the respective lists
        x1, y1 = nearest_points[0]
        x2, y2 = nearest_points[1]
        shortest_points_x.append([x1, x2])
        shortest_points_y.append([y1, y2])
    
    return shortest_distances, shortest_points_x, shortest_points_y

def findLongestDistanceContour(window, contoured_frames):
    longest_distances = []
    longest_points_x = []  # List to store x coordinates of the points
    longest_points_y = []  # List to store y coordinates of the points

    for frame in contoured_frames:
        polygon = Polygon([(x, y) for x, y in zip(window.lumen[0][frame], window.lumen[1][frame])])

        # Get the exterior coordinates of the polygon
        exterior_coords = polygon.exterior.coords

        max_distance = 0 
        farthest_points = None

        for point1, point2 in combinations(exterior_coords, 2):
            distance = math.dist(point1, point2)
            if distance > max_distance:
                max_distance = distance
                farthest_points = (point1, point2)

        longest_distances.append(max_distance * window.resolution)
        
        # Separate x and y coordinates and append to the respective lists
        x1, y1 = farthest_points[0]
        x2, y2 = farthest_points[1]
        longest_points_x.append([x1, x2])
        longest_points_y.append([y1, y2])
    
    return longest_distances, longest_points_x, longest_points_y


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

def plotContoursWithMetrics(window, contoured_frames, centroid_x, centroid_y, lumen_area, shortest_distances, longest_distances):
    """Plot contours and annotate with metrics"""
    longest_distances, longest_x, longest_y = findLongestDistanceContour(window, contoured_frames)
    shortest_distances, shortest_x, shortest_y = findShortestDistanceContour(window, contoured_frames)
    first_third = int(len(contoured_frames) * 0.25)
    second_third = int(len(contoured_frames) * 0.5)
    third_third = int(len(contoured_frames) * 0.75)
    indices_to_plot = [first_third, second_third, third_third]
    frames_to_plot = [contoured_frames[index] for index in indices_to_plot]

    for index, frame in enumerate(frames_to_plot):
        plt.figure(figsize=(6, 6))
        plt.plot(window.lumen[0][frame], window.lumen[1][frame], '-g', linewidth=2, label='Contour')
        plt.plot(centroid_x[index], centroid_y[index], 'ro', markersize=8, label='Centroid')
        plt.plot(longest_x[index][0], longest_y[index][0], 'bo', markersize=8, label='Farthest Point 1')
        plt.plot(longest_x[index][1], longest_y[index][1], 'bo', markersize=8, label='Farthest Point 2')
        plt.plot(shortest_x[index], shortest_y[index], 'yo', markersize=8, label='Shortest Point')


        # Annotate with shortest and longest distances
        plt.annotate(f'Shortest Distance: {shortest_distances[index]:.2f} mm',
                     xy=(centroid_x[index], centroid_y[index]), xycoords='data',
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3,rad=.2"))

        plt.annotate(f'Longest Distance: {longest_distances[index]:.2f} mm',
                     xy=(centroid_x[index], centroid_y[index]), xycoords='data',
                     xytext=(10, -30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3,rad=-.2"))
        
        plt.annotate(f'Lumen Area: {lumen_area[index]:.2f} mm\N{SUPERSCRIPT TWO}',
                     xy=(centroid_x[index], centroid_y[index]), xycoords='data',
                     xytext=(10, 0), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3,rad=0"))

        plt.title(f'Frame {frame}')
        plt.legend(loc='upper right')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.tight_layout()
        plt.show()