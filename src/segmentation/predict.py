import cv2
import numpy as np
from loguru import logger
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt
from gui.popup_windows.message_boxes import ErrorMessage
import gc


class Predict:

    def __init__(self, main_window, config=None) -> None:
        self.main_window = main_window
        config = main_window.config if config is None else config
        self.model_file = config.segmentation.model_file
        self.model_fold = config.segmentation.model_fold
        self.model_name = config.segmentation.model_name
        self.normalize = config.segmentation.normalize
        self.batch_size = config.segmentation.batch_size
        self.conserve_memory = config.segmentation.conserve_memory
        self.images = None

    def __call__(self, images, lower_limit, upper_limit) -> None:
        self.images = images
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.normalisation(self.normalize)
        mask = self.inference()

        return mask

    def normalisation(self, do_it_or_not: bool):
        """Min-max normalisation of the images"""
        if do_it_or_not:
            self.images = (self.images - self.images.min(axis=(1, 2), keepdims=True)) / (
                    self.images.max(axis=(1, 2), keepdims=True) - self.images.min(axis=(1, 2), keepdims=True)
            )

    def inference(self):
        if "nnUNetTrainer" not in self.model_file:
            import tensorflow as tf
            custom_objects = {'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
            model = tf.keras.models.load_model(self.model_file, custom_objects=custom_objects, compile=False)

            self.check_input_shape(model.input_shape, )
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
                    pred = model(self.images[frame: frame + self.batch_size, :, :], training=False)
                    mask[frame: frame + self.batch_size, :, :] = np.array(pred)[0, :, :, :, 0]
                    if progress is not None and progress.wasCanceled():
                        progress.close()
                        return None
                if progress is not None:
                    progress.close()
            else:
                prediction = model.predict(
                    self.images[self.lower_limit: self.upper_limit, :, :], batch_size=self.batch_size, verbose=1
                )
                mask[self.lower_limit: self.upper_limit, :, :] = np.array(prediction)[0, :, :, :, 0]
        else:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            seg_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            # initializes the network architecture, loads the checkpoint
            seg_predictor.initialize_from_trained_model_folder(
                self.model_file,
                use_folds=(self.model_fold,),
                checkpoint_name="checkpoint_final.pth",
            )
            print(f"Shape: {self.images.shape}")
            # mask = seg_predictor.predict_from_list_of_npy_arrays([img[None, None, ...] for img in self.images],
            #                                               segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
            #                                               properties_or_list_of_properties=[dict(spacing=[1, 1, 1]) for _ in self.images],
            #                                               truncated_ofname=None,
            #                                               num_processes=1)
            mask = seg_predictor.predict_single_npy_array(self.images[None, ...].astype(np.float32),
                                                          image_properties=dict(spacing=[1, 1, 1]))
            print(f"mask shape: {mask.shape}")
        return mask

    @staticmethod
    def resize_with_crop_or_pad_cv2(img: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Resize an image (NumPy array H×W or H×W×C) with cropping or padding to maintain aspect ratio,
        using OpenCV and NumPy only.
        """
        # ensure we have H×W×C
        if img.ndim == 2:
            img = img[:, :, None]

        original_height, original_width = img.shape[:2]
        target_ratio = target_width / target_height
        original_ratio   = original_width / original_height

        # first, scale so that one dimension matches
        if original_ratio > target_ratio:
            scale = target_height / original_height
        else:
            scale = target_width / original_width

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # now either crop or pad to get exactly target WxH
        top = bottom = left = right = 0

        # horizontal adjustment
        if new_width > target_width:
            # crop width
            left = (new_width - target_width) // 2
            right = left + target_width
            cropped = resized[:, left:right, :]
        else:
            # pad width
            pad = target_width - new_width
            left = pad // 2
            right = pad - left
            cropped = cv2.copyMakeBorder(resized, 0, 0, left, right,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

        # vertical adjustment
        if cropped.shape[0] > target_height:
            top = (cropped.shape[0] - target_height) // 2
            cropped = cropped[top:top+target_height, :, :]
        else:
            pad = target_height - cropped.shape[0]
            top = pad // 2
            bottom = pad - top
            cropped = cv2.copyMakeBorder(cropped, top, bottom, 0, 0,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

        # if originally single‐channel, squeeze back
        if cropped.shape[2] == 1:
            cropped = cropped[:, :, 0]

        return cropped

    def check_input_shape(self, input_shape, batch_size=16):
        """
        Check if the input shape of the model matches the shape of the images.
        model.input_shape

        """
        logger.info(f"Input shape: {self.images.shape}")
        if input_shape[1] != self.images.shape[1] or input_shape[2] != self.images.shape[2]:
            logger.warning("Reshaping the images to match the model input shape.")

            # Ensure images are in float32 to save memory
            self.images = self.images.astype(np.float32)

            # Process images in batches to reduce memory usage
            num_images = self.images.shape[0]
            reshaped_images = []

            for start in range(0, num_images, batch_size):
                end = min(start + batch_size, num_images)
                batch = self.images[start:end]
                batch = np.expand_dims(batch, axis=-1)
                batch = [self.resize_with_crop_or_pad_pil(img, input_shape[1], input_shape[2]) for img in batch]
                # batch = resize_with_crop_or_pad(batch, model.input_shape[1], model.input_shape[2])
                batch = np.squeeze(batch, axis=-1)
                reshaped_images.append(batch)

                # Explicitly call garbage collection
                gc.collect()

            self.images = np.concatenate(reshaped_images, axis=0)
            gc.collect()
