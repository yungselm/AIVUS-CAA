import xml.etree.ElementTree as et
import os
import datetime
import numpy as np
import matplotlib.path as mplPath
from loguru import logger
from skimage import measure

from version import version_file_str


def label_contours(image):
    """generate contours for labels"""
    # Find contours at a constant value
    contours = measure.find_contours(image)
    lumen = []
    for contour in contours:
        lumen.append(np.array((contour[:, 0], contour[:, 1])))

    return lumen


def keep_largest_contour(contours, image_shape):
    # this function returns the largest contour (num of points) as a numpy array
    max_length = 0
    keep_contour = [[], []]
    for contour in contours:
        if keep_valid_contour(contour, image_shape):
            if len(contour[0]) > max_length:
                keep_contour = [list(contour[1, :]), list(contour[0, :])]
                max_length = len(contour[0])

    return keep_contour


def keep_valid_contour(contour, image_shape):
    # this function check that the contour is valid if the image centroid is contained within the mask region
    bbPath = mplPath.Path(np.transpose(contour))
    centroid = [image_shape[0] // 2, image_shape[1] // 2]
    return bbPath.contains_point(centroid)


def get_contours(preds, image_shape):
    """Extracts contours from masked images. Returns x and y coodinates"""
    # get contours for each image
    lumen_pred = [[], []]
    # convert contours to x and y points
    for frame in range(preds.shape[0]):
        if np.any(preds[frame, :, :] == 1):
            lumen = label_contours(preds[frame, :, :])
            # return the contour with the largest number of points
            keep_lumen_x, keep_lumen_y = keep_largest_contour(lumen, image_shape)
            lumen_pred[0].append(keep_lumen_x)
            lumen_pred[1].append(keep_lumen_y)
        else:
            lumen_pred[0].append([])
            lumen_pred[1].append([])

    return lumen_pred


def write_xml(x, y, dims, resolution, speed, plaque_frames, phases, out_path):
    """Write an xml file of contour data

    Args:
        x: list, where alternating entries are lists of lumen/plaque x points
        y: list, where alternating entries are lists of lumen/plaque y points
        dims: list, where entries are image height, width and number of images
        resolution: float, image resolution (mm)
        speed: float, speed of pullback mm/s
        pname: string: name of the output file
    Returns:
        None
    """

    num_frames = dims[0]
    root = et.Element('AnalysisState')
    analyzedfilename = et.SubElement(root, 'AnalyzedFileName')
    analyzedfilename.text = 'FILE0000'
    analyzedfilenamefullpath = et.SubElement(root, 'AnalyzedFileNameFullPath')
    analyzedfilenamefullpath.text = 'D:\CASE0000\FILE0000'
    username = et.SubElement(root, 'UserName')
    username.text = 'ICViewAdmin'
    computername = et.SubElement(root, 'ComputerName')
    computername.text = 'USER-3BF85F9281'
    softwareversion = et.SubElement(root, 'SoftwareVersion')
    softwareversion.text = '4.0.27'
    screenresolution = et.SubElement(root, 'ScreenResolution')
    screenresolution.text = '1600 x 900'
    date = et.SubElement(root, 'Date')
    date.text = datetime.datetime.now().strftime('%d%b%Y %H:%M:%S')
    timezone = et.SubElement(root, 'TimeZone')
    timezone.text = 'GMT-300 min'
    demographics = et.SubElement(root, 'Demographics')
    patientname = et.SubElement(demographics, 'PatientName')
    patientname.text = os.path.basename(out_path)
    patientid = et.SubElement(demographics, 'PatientID')
    patientid.text = os.path.basename(out_path)

    imagestate = et.SubElement(root, 'ImageState')
    xdim = et.SubElement(imagestate, 'Xdim')
    xdim.text = str(dims[1])
    ydim = et.SubElement(imagestate, 'Ydim')
    ydim.text = str(dims[2])
    numberofframes = et.SubElement(imagestate, 'NumberOfFrames')
    numberofframes.text = str(num_frames)
    firstframeloaded = et.SubElement(imagestate, 'FirstFrameLoaded')
    firstframeloaded.text = str(0)
    lastframeloaded = et.SubElement(imagestate, 'LastFrameLoaded')
    lastframeloaded.text = str(num_frames - 1)
    stride = et.SubElement(imagestate, 'Stride')
    stride.text = str(1)

    imagecalibration = et.SubElement(root, 'ImageCalibration')
    xcalibration = et.SubElement(imagecalibration, 'XCalibration')
    xcalibration.text = str(resolution)
    ycalibration = et.SubElement(imagecalibration, 'YCalibration')
    ycalibration.text = str(resolution)
    acqrateinfps = et.SubElement(imagecalibration, 'AcqRateInFPS')
    acqrateinfps.text = str(133.0)
    pullbackspeed = et.SubElement(imagecalibration, 'PullbackSpeed')
    pullbackspeed.text = str(speed)

    brightnesssetting = et.SubElement(root, 'BrightnessSetting')
    brightnesssetting.text = str(50)
    contrastsetting = et.SubElement(root, 'ContrastSetting')
    contrastsetting.text = str(50)
    freestepping = et.SubElement(root, 'FreeStepping')
    freestepping.text = 'FALSE'
    steppinginterval = et.SubElement(root, 'SteppingInterval')
    steppinginterval.text = str(1)
    volumehasbeencomputed = et.SubElement(root, 'VolumeHasBeenComputed')
    volumehasbeencomputed.text = 'FALSE'

    framestate = et.SubElement(root, 'FrameState')
    imagerelativepoints = et.SubElement(framestate, 'ImageRelativePoints')
    imagerelativepoints.text = 'TRUE'
    xoffset = et.SubElement(framestate, 'Xoffset')
    xoffset.text = str(109)
    yoffset = et.SubElement(framestate, 'Yoffset')
    yoffset.text = str(3)
    for frame in range(num_frames):
        fm = et.SubElement(framestate, 'Fm')
        num = et.SubElement(fm, 'Num')
        num.text = str(frame)
        plaque = et.SubElement(fm, 'Plaque')
        phase = et.SubElement(fm, 'Phase')
        try:
            plaque.text = plaque_frames[frame]
        except IndexError:  # old contour files may not have phases attr
            plaque.text = '0'
        try:
            phase.text = phases[frame]
        except IndexError:  # old contour files may not have phases attr
            phase.text = '-'

        try:
            ctr = et.SubElement(fm, 'Ctr')
            npts = et.SubElement(ctr, 'Npts')
            npts.text = str(len(x[frame]))
            type = et.SubElement(ctr, 'Type')
            type.text = 'L'
            handdrawn = et.SubElement(ctr, 'HandDrawn')
            handdrawn.text = 'T'
            # iterative over the points in each contour
            for k in range(len(x[frame])):
                p = et.SubElement(ctr, 'p')
                p.text = str(int(x[frame][k])) + ',' + str(int(y[frame][k]))
        except IndexError:
            pass

    tree = et.ElementTree(root)
    tree.write(out_path + f'_contours_{version_file_str}.xml')
