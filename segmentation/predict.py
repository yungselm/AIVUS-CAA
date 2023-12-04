import numpy as np
import tensorflow as tf
from loguru import logger
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt


class Predict:
    def __init__(self, main_window) -> None:
        self.main_window = main_window
        self.model_file = main_window.config.segmentation.model_file
        self.batch_size = main_window.config.segmentation.batch_size
        self.conserve_memory = main_window.config.segmentation.conserve_memory

    def __call__(self, images, lower_limit, upper_limit) -> None:
        self.images = images
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.normalisation()
        mask = self.inference()

        return mask

    def normalisation(self):
        """Min-max normalisation of the images"""
        self.images = (self.images - self.images.max(axis=(1, 2), keepdims=True)) / (
            self.images.max(axis=(1, 2), keepdims=True) - self.images.min(axis=(1, 2), keepdims=True)
        )

    def inference(self):
        custom_objects = {'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
        model = tf.keras.models.load_model(self.model_file, custom_objects=custom_objects, compile=False)
        mask = np.zeros_like(self.images)

        if self.conserve_memory:
            progress = QProgressDialog(self.main_window)
            progress.setWindowFlags(Qt.Dialog)
            progress.setModal(True)
            progress.setMinimum(self.lower_limit)
            progress.setMaximum(self.upper_limit)
            progress.setMinimumDuration(1000)
            progress.resize(500, 100)
            progress.setWindowTitle('Automatic segmentation')
            progress.setLabelText(f'Please wait, segmenting frames {self.lower_limit} to {self.upper_limit}...')
            progress.show()

            for frame in range(self.lower_limit, self.upper_limit, self.batch_size):
                progress.setValue(frame)
                # calling model() instead of model.predict() leads to smaller memory leak
                pred = model(self.images[frame : frame + self.batch_size, :, :], training=False)
                mask[frame : frame + self.batch_size, :, :] = np.array(pred)[0, :, :, :, 0]
                if progress.wasCanceled():
                    progress.close()
                    return None
            progress.close()
        else:
            prediction = model.predict(
                self.images[self.lower_limit : self.upper_limit, :, :], batch_size=self.batch_size, verbose=0
            )
            mask[self.lower_limit : self.upper_limit, :, :] = np.array(prediction)[0, :, :, :, 0]
            self.main_window.successMessage('Automatic segmentation')

        return mask
