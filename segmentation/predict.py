import numpy as np
import tensorflow as tf
from loguru import logger
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt


class Predict:
    def __init__(self, main_window, config=None) -> None:
        self.main_window = main_window
        config = main_window.config if config is None else config
        self.model_file = config.segmentation.model_file
        self.batch_size = config.segmentation.batch_size
        self.conserve_memory = config.segmentation.conserve_memory

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

        self.check_input_shape(model)
        mask = np.zeros_like(self.images)

        if self.conserve_memory:
            if self.main_window is not None:
                progress = QProgressDialog(self.main_window)
                progress.setWindowFlags(Qt.Dialog)
                progress.setModal(True)
                progress.setMinimum(self.lower_limit)
                progress.setMaximum(self.upper_limit)
                progress.setMinimumDuration(1000)
                progress.resize(500, 100)
                progress.setWindowTitle('Automatic segmentation')
                progress.setLabelText(
                    f'Please wait, segmenting frames {self.lower_limit + 1} to {self.upper_limit + 1}...'
                )
                progress.show()
            else:
                progress = None

            for frame in range(self.lower_limit, self.upper_limit, self.batch_size):
                if progress is not None:
                    progress.setValue(frame)
                # calling model() instead of model.predict() leads to smaller memory leak
                pred = model(self.images[frame : frame + self.batch_size, :, :], training=False)
                mask[frame : frame + self.batch_size, :, :] = np.array(pred)[0, :, :, :, 0]
                if progress is not None and progress.wasCanceled():
                    progress.close()
                    return None
            if progress is not None:
                progress.close()
        else:
            prediction = model.predict(
                self.images[self.lower_limit : self.upper_limit, :, :], batch_size=self.batch_size, verbose=1
            )
            mask[self.lower_limit : self.upper_limit, :, :] = np.array(prediction)[0, :, :, :, 0]

        return mask

    def check_input_shape(self, model):
        """Check if the input shape of the model matches the shape of the images."""
        logger.info(f"Input shape: {self.images.shape}")
        if model.input_shape[1] != self.images.shape[1] or model.input_shape[2] != self.images.shape[2]:
            logger.warning("Reshaping the images to match the model input shape.")
            self.images = np.expand_dims(self.images, axis=-1)
            self.images = tf.image.resize_with_crop_or_pad(self.images, model.input_shape[1], model.input_shape[2])
            self.images = np.squeeze(self.images, axis=-1)
