import os

from loguru import logger
import SimpleITK as sitk
from monailabel.transform.writer import write_nifti
from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtCore import Qt

from input_output.contours import contoursToMask


def save_as_nifti(window):
    if not window.image:
        warning = QErrorMessage(window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot save as NIfTI before reading DICOM file')
        warning.exec_()
        return

    contoured_frames = [
        frame for frame in range(window.numberOfFrames) if window.lumen[0][frame] or window.plaque[0][frame]
    ]  # find frames with contours (no need to save the others)

    if contoured_frames:
        file_name = os.path.splitext(os.path.basename(window.file_name))[0]  # remove file extension
        out_path = os.path.join(os.path.dirname(window.file_name), 'niftis', file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mask = contoursToMask(window.images[contoured_frames], contoured_frames, window.lumen, window.plaque)
        
        for i, frame in enumerate(contoured_frames):  # save individual frames as NIfTI
            sitk.WriteImage(sitk.GetImageFromArray(mask[i, :, :]), f'{out_path}_frame_{i}_seg.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(window.images[frame, :, :]), f'{out_path}_frame_{i}_img.nii.gz')

        # save entire stack as NIfTI
        sitk.WriteImage(sitk.GetImageFromArray(mask), f'{out_path}_seg.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(window.images[contoured_frames, :, :]), f'{out_path}_img.nii.gz')
