import os

from loguru import logger
import SimpleITK as sitk
from PyQt5.QtWidgets import QErrorMessage, QProgressDialog, QApplication
from PyQt5.QtCore import Qt

from input_output.contours import contours_to_mask


def save_as_nifti(main_window):
    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot save as NIfTi before reading DICOM file')
        warning.exec_()
        return

    out_path = f'{main_window.config.save.nifti_dir}_{main_window.config.save.save_niftis}_frames'
    if main_window.config.save.save_niftis == 'contoured':
        frames_to_save = [
            frame for frame in range(main_window.metadata['number_of_frames']) if main_window.data['lumen'][0][frame]
        ]  # find frames with contours (no need to save the others)
    elif main_window.config.save.save_niftis == 'all':
        frames_to_save = range(main_window.metadata['number_of_frames'])
    else:
        return  # nothing to save

    if frames_to_save:
        file_name = os.path.splitext(os.path.basename(main_window.file_name))[0]  # remove file extension
        os.makedirs(out_path, exist_ok=True)
        mask = contours_to_mask(main_window.images[frames_to_save], frames_to_save, main_window.data['lumen'])

        progress = QProgressDialog()
        progress.setWindowFlags(Qt.Dialog)
        progress.setModal(True)
        progress.setMinimum(0)
        progress_max = len(frames_to_save) * main_window.config.save.save_2d + main_window.config.save.save_3d
        progress.setMaximum(progress_max)
        progress.resize(500, 100)
        progress.setValue(0)
        progress.setValue(1)
        progress.setValue(0)  # trick to make progress bar appear
        progress.setWindowTitle("Saving frames as NIfTi files...")
        progress.show()

        if main_window.config.save.save_2d:
            for i, frame in enumerate(frames_to_save):  # save individual frames as NIfTi
                progress.setValue(i)
                QApplication.processEvents()
                if progress.wasCanceled():
                    break
                if main_window.data['lumen'][0][frame]:  # only save mask if contour exists
                    sitk.WriteImage(
                        sitk.GetImageFromArray(mask[i, :, :]),
                        os.path.join(out_path, f'{file_name}_frame_{i}_seg.nii.gz'),
                    )
                sitk.WriteImage(
                    sitk.GetImageFromArray(main_window.images[frame, :, :]),
                    os.path.join(out_path, f'{file_name}_frame_{i}_img.nii.gz'),
                )
        if main_window.config.save.save_3d:
            if any(main_window.data['lumen'][0]):  # only save mask if any contour exists
                sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(out_path, f'{file_name}_seg.nii.gz'))
            sitk.WriteImage(
                sitk.GetImageFromArray(main_window.images), os.path.join(out_path, f'{file_name}_img.nii.gz')
            )
            progress.setValue(len(frames_to_save) * main_window.config.save.save_2d + 1)
            QApplication.processEvents()

        progress.close()
