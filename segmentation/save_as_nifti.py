import os

from loguru import logger
import SimpleITK as sitk
from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtCore import Qt

from input_output.contours import contoursToMask


def save_as_nifti(main_window):
    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot save as NIfTI before reading DICOM file')
        warning.exec_()
        return

    contoured_frames = [
        frame for frame in range(main_window.metadata['number_of_frames']) if main_window.data['lumen'][0][frame]
    ]  # find frames with contours (no need to save the others)

    if contoured_frames:
        file_name = os.path.splitext(os.path.basename(main_window.file_name))[0]  # remove file extension
        out_path = os.path.join(os.path.dirname(main_window.file_name), 'niftis', file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mask = contoursToMask(main_window.images[contoured_frames], contoured_frames, main_window.data['lumen'])

        for i, frame in enumerate(contoured_frames):  # save individual frames as NIfTI
            sitk.WriteImage(sitk.GetImageFromArray(mask[i, :, :]), f'{out_path}_frame_{i}_seg.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(main_window.images[frame, :, :]), f'{out_path}_frame_{i}_img.nii.gz')

        # save entire stack as NIfTI
        # sitk.WriteImage(sitk.GetImageFromArray(mask), f'{out_path}_seg.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(main_window.images[contoured_frames, :, :]), f'{out_path}_img.nii.gz')
