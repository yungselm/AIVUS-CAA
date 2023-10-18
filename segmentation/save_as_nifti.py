import os

from loguru import logger
import SimpleITK as sitk
from PyQt5.QtWidgets import QErrorMessage, QProgressDialog
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
        out_path = main_window.config.nifti_dir
        os.makedirs(out_path, exist_ok=True)
        mask = contoursToMask(main_window.images[contoured_frames], contoured_frames, main_window.data['lumen'])

        progress = QProgressDialog()
        progress.setWindowFlags(Qt.Dialog)
        progress.setModal(True)
        progress.setMinimum(0)
        progress.setMaximum(len(contoured_frames))
        progress.resize(500, 100)
        progress.setValue(0)
        progress.setValue(1)
        progress.setValue(0)  # trick to make progress bar appear
        progress.setWindowTitle("Saving contoured frames as NIfTI files...")
        progress.show()

        for i, frame in enumerate(contoured_frames):  # save individual frames as NIfTI
            progress.setValue(i)
            if progress.wasCanceled():
                break
            
            sitk.WriteImage(
                sitk.GetImageFromArray(mask[i, :, :]), os.path.join(out_path, f'{file_name}_frame_{i}_seg.nii.gz')
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(main_window.images[frame, :, :]),
                os.path.join(out_path, f'{file_name}_frame_{i}_img.nii.gz'),
            )

        progress.close()
