import xml.etree.ElementTree as ET


def read_xml(main_window, path, frames=[]):
    tree = ET.parse(path)  # current version
    root = tree.getroot()
    root.attrib
    lumen_points = []
    frame_list = []
    phases = []
    lumen = {}

    for child in root:
        for image_state in child.iter('ImageState'):
            dim_z = image_state.find('NumberOfFrames').text
            if not frames:
                frames = range(int(dim_z))

        for image_calibration in child.iter('ImageCalibration'):
            res_x = image_calibration.find('XCalibration').text

        for _ in child.iter('FrameState'):
            for frame in child.iter('Fm'):
                frame_number = int(frame.find('Num').text)
                lumen_subpoints = []
                if frame_number in frames:
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

    main_window.data['lumen'] = split_x_y(lumen_points)
    main_window.data['phases'] = phases
    main_window.metadata['resolution'] = res_x


def split_x_y(points):
    """Splits comma separated points into separate x and y lists"""

    points_x = []
    points_y = []
    for i in range(0, len(points)):
        points_x.append(map(lambda x: int(x.split(',')[0]), points[i]))
        points_y.append(map(lambda x: int(x.split(',')[1]), points[i]))

    return points_x, points_y
