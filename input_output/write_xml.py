import xml.etree.ElementTree as et
import os
import datetime

from version import version_file_str


def write_xml(x, y, dims, resolution, speed, plaque_frames, phases, out_path):
    """Write an xml file of contour data

    Args:
        x: list, where alternating entries are lists of lumen x points
        y: list, where alternating entries are lists of lumen y points
        dims: list, where entries are image height, width and number of images
        resolution: float, image resolution (mm)
        speed: float, speed of pullback mm/s
        pname: string: name of the output file
    Returns:
        None
    """

    num_frames = dims[0]
    root = et.Element('AnalysisState')
    analysed_filename = et.SubElement(root, 'AnalyzedFileName')
    analysed_filename.text = 'FILE0000'
    analysed_path = et.SubElement(root, 'AnalyzedFileNameFullPath')
    analysed_path.text = 'D:\CASE0000\FILE0000'
    user_name = et.SubElement(root, 'UserName')
    user_name.text = 'ICViewAdmin'
    computer_name = et.SubElement(root, 'ComputerName')
    computer_name.text = 'USER-3BF85F9281'
    software_version = et.SubElement(root, 'SoftwareVersion')
    software_version.text = '4.0.27'
    screen_resolution = et.SubElement(root, 'ScreenResolution')
    screen_resolution.text = '1600 x 900'
    date = et.SubElement(root, 'Date')
    date.text = datetime.datetime.now().strftime('%d%b%Y %H:%M:%S')
    time_zone = et.SubElement(root, 'TimeZone')
    time_zone.text = 'GMT-300 min'
    demographics = et.SubElement(root, 'Demographics')
    patient_name = et.SubElement(demographics, 'PatientName')
    patient_name.text = os.path.basename(out_path)
    patient_id = et.SubElement(demographics, 'PatientID')
    patient_id.text = os.path.basename(out_path)

    image_state = et.SubElement(root, 'ImageState')
    dim_x = et.SubElement(image_state, 'Xdim')
    dim_x.text = str(dims[1])
    dim_y = et.SubElement(image_state, 'Ydim')
    dim_y.text = str(dims[2])
    number_of_frames = et.SubElement(image_state, 'NumberOfFrames')
    number_of_frames.text = str(num_frames)
    first_frame_loaded = et.SubElement(image_state, 'FirstFrameLoaded')
    first_frame_loaded.text = str(0)
    last_frame_loaded = et.SubElement(image_state, 'LastFrameLoaded')
    last_frame_loaded.text = str(num_frames - 1)
    stride = et.SubElement(image_state, 'Stride')
    stride.text = str(1)

    image_calibration = et.SubElement(root, 'ImageCalibration')
    calibration_x = et.SubElement(image_calibration, 'XCalibration')
    calibration_x.text = str(resolution)
    calibration_y = et.SubElement(image_calibration, 'YCalibration')
    calibration_y.text = str(resolution)
    acq_rate_fps = et.SubElement(image_calibration, 'AcqRateInFPS')
    acq_rate_fps.text = str(133.0)
    pullback_speed = et.SubElement(image_calibration, 'PullbackSpeed')
    pullback_speed.text = str(speed)

    brightness_setting = et.SubElement(root, 'BrightnessSetting')
    brightness_setting.text = str(50)
    contrast_setting = et.SubElement(root, 'ContrastSetting')
    contrast_setting.text = str(50)
    free_stepping = et.SubElement(root, 'FreeStepping')
    free_stepping.text = 'FALSE'
    stepping_interval = et.SubElement(root, 'SteppingInterval')
    stepping_interval.text = str(1)
    volume_computed = et.SubElement(root, 'VolumeHasBeenComputed')
    volume_computed.text = 'FALSE'

    frame_state = et.SubElement(root, 'FrameState')
    img_rel_points = et.SubElement(frame_state, 'ImageRelativePoints')
    img_rel_points.text = 'TRUE'
    offset_x = et.SubElement(frame_state, 'Xoffset')
    offset_x.text = str(109)
    offset_y = et.SubElement(frame_state, 'Yoffset')
    offset_y.text = str(3)
    for frame_index in range(num_frames):
        frame = et.SubElement(frame_state, 'Fm')
        frame_number = et.SubElement(frame, 'Num')
        frame_number.text = str(frame_index)
        phase = et.SubElement(frame, 'Phase')
        try:
            phase.text = phases[frame_index]
        except IndexError:  # old contour files may not have phases attr
            phase.text = '-'

        try:
            contour = et.SubElement(frame, 'Ctr')
            num_points = et.SubElement(contour, 'Npts')
            num_points.text = str(len(x[frame_index]))
            type = et.SubElement(contour, 'Type')
            type.text = 'L'
            hand_drawn = et.SubElement(contour, 'HandDrawn')
            hand_drawn.text = 'T'
            # iterative over the points in each contour
            for k in range(len(x[frame_index])):
                p = et.SubElement(contour, 'p')
                p.text = str(int(x[frame_index][k])) + ',' + str(int(y[frame_index][k]))
        except IndexError:
            pass

    tree = et.ElementTree(root)
    tree.write(out_path + f'_contours_{version_file_str}.xml')
