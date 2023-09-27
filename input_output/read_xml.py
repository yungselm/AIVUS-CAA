import glob

import xml.etree.ElementTree as ET
from loguru import logger

from version import version_file_str


def splitxy(points):
    """Splits comma separated points into separate x and y lists"""

    pointsX = []
    pointsY = []
    for i in range(0, len(points)):
        pointsX.append(map(lambda x: int(x.split(',')[0]), points[i]))
        pointsY.append(map(lambda x: int(x.split(',')[1]), points[i]))

    return pointsX, pointsY


def read(path, frames=[]):
    """Reads xml file from the specified path.

    Args:
        path: str, path to the .xml file (must be in echoplaque format)
        frames: list, frames that should be included, if empty all are included
    Returns:
        (Lx, Ly): tuple, x and y lumen contours
        (Vx, Vy): tuple, x and y plaque contours
        [xres, yres]: list, x and y pixel spacing
        framelist: list, frames with contours
    """
    xml_files = glob.glob(f'{path}_contours_*.xml')
    xml_legacy_file = glob.glob(f'{path}_contours.xml')  # legacy file without version number

    if xml_files:
        newest_xml = max(xml_files)  # find file with most recent version
    else:
        newest_xml = xml_legacy_file[0]

    logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_xml}')
    
    tree = ET.parse(newest_xml)  # current version
    root = tree.getroot()
    root.attrib
    lumen_points = []
    framelist = []
    plaque_frames = []
    phases = []
    lumen = {}

    for child in root:
        for imageState in child.iter('ImageState'):
            xdim = imageState.find('Xdim').text
            ydim = imageState.find('Ydim').text
            zdim = imageState.find('NumberOfFrames').text
            if not frames:
                frames = range(int(zdim))

        for imageCalibration in child.iter('ImageCalibration'):
            xres = imageCalibration.find('XCalibration').text
            yres = imageCalibration.find('YCalibration').text

        for _ in child.iter('FrameState'):
            for frame in child.iter('Fm'):
                frameNo = int(frame.find('Num').text)
                lumen_subpoints = []
                if frameNo in frames:
                    try:
                        plaque_frames.append(frame.find('Plaque').text)
                    except AttributeError:  # old contour files may not have plaque attribute
                        plaque_frames.append('0')
                    try:
                        phase = frame.find('Phase').text
                        phase = '-' if phase is None else phase
                    except AttributeError:  # old contour files may not have phase attribute
                        phase = '-'
                    phases.append(phase)
                    for pts in frame.iter('Ctr'):
                        framelist.append(frameNo)
                        for child in pts:
                            if child.tag == 'Type':
                                if child.text == 'L':
                                    contour = 'L'
                            # add each point
                            elif child.tag == 'p':
                                if contour == 'L':
                                    lumen_subpoints.append(child.text)
                    lumen_points.append(lumen_subpoints)
                    lumen[frameNo] = lumen_subpoints

    Lx, Ly = splitxy(lumen_points)

    # return unique frames as we have entry for each inner and outer contour
    framelist = list(sorted(set(map(int, framelist))))

    return (Lx, Ly), [xres, yres], [xdim, ydim, zdim], plaque_frames, phases
