import xml.etree.ElementTree as ET


def split_x_y(points):
    """Splits comma separated points into separate x and y lists"""

    points_x = []
    points_y = []
    for i in range(0, len(points)):
        points_x.append(map(lambda x: int(x.split(',')[0]), points[i]))
        points_y.append(map(lambda x: int(x.split(',')[1]), points[i]))

    return points_x, points_y


def read(path, frames=[]):
    tree = ET.parse(path)  # current version
    root = tree.getroot()
    root.attrib
    lumen_points = []
    frame_list = []
    plaque_frames = []
    phases = []
    lumen = {}

    for child in root:
        for image_state in child.iter('ImageState'):
            dim_x = image_state.find('Xdim').text
            dim_y = image_state.find('Ydim').text
            dim_z = image_state.find('NumberOfFrames').text
            if not frames:
                frames = range(int(dim_z))

        for image_calibration in child.iter('ImageCalibration'):
            res_x = image_calibration.find('XCalibration').text
            res_y = image_calibration.find('YCalibration').text

        for _ in child.iter('FrameState'):
            for frame in child.iter('Fm'):
                frame_number = int(frame.find('Num').text)
                lumen_subpoints = []
                if frame_number in frames:
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
                        frame_list.append(frame_number)
                        for child in pts:
                            if child.tag == 'Type':
                                if child.text == 'L':
                                    contour = 'L'
                            # add each point
                            elif child.tag == 'p':
                                if contour == 'L':
                                    lumen_subpoints.append(child.text)
                    lumen_points.append(lumen_subpoints)
                    lumen[frame_number] = lumen_subpoints

    Lx, Ly = split_x_y(lumen_points)

    # return unique frames as we have entry for each inner and outer contour
    frame_list = list(sorted(set(map(int, frame_list))))

    return (Lx, Ly), [res_x, res_y], [dim_x, dim_y, dim_z], plaque_frames, phases
