import os

import numpy as np
from loguru import logger
from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtCore import Qt
from shapely.geometry import Polygon


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

    window.lumen, window.plaque = window.wid.getData()
    contoured_frames = [
        frame for frame in range(window.numberOfFrames) if window.lumen[0][frame] or window.plaque[0][frame]
    ]
    lumen_area, centroid_x, centroid_y = computeContourMetrics(window, contoured_frames)

    f = open(os.path.splitext(window.file_name)[0] + "_report.txt", "w")
    f.write(
        "Frame\tPosition (mm)\tLumen area (mm\N{SUPERSCRIPT TWO})"
        "\tCentroid x (px)\tCentroid y (px)\tPlaque\tPhase\n"
    )

    for index, frame in enumerate(contoured_frames):
        f.write(
            "{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\n".format(
                frame,
                window.pullbackLength[frame],
                lumen_area[index],
                centroid_x[index],
                centroid_y[index],
                window.plaque_frames[frame],
                window.phases[frame],
            )
        )
    f.close()

    window.successMessage("Write report")


def computeContourMetrics(window, contoured_frames):
    """Computes lumen area, plaque area and plaque burden from contours"""

    lumen_area = np.zeros(len(contoured_frames))
    centroid_0x = np.zeros(len(contoured_frames))
    centroid_0y = np.zeros(len(contoured_frames))
    for index, frame in enumerate(contoured_frames):
        if window.lumen[0][frame]:
            lumen_area[index] = contourArea(window.lumen[0][frame], window.lumen[1][frame]) * window.resolution**2
            centroid_x = centroidPolygonSimple(window.lumen[0][frame], window.lumen[1][frame])[0]  
            centroid_y = centroidPolygonSimple(window.lumen[0][frame], window.lumen[1][frame])[1]
            polygon = Polygon([(x, y) for x, y in zip(window.lumen[0][frame], window.lumen[1][frame])])
            centroid_xx = polygon.centroid.x
            centroid_yy = polygon.centroid.y 
            centroid_0x[index] = centroid_x - centroid_xx
            centroid_0y[index] = centroid_y - centroid_yy


    return (lumen_area, centroid_0x, centroid_0y)


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

    

