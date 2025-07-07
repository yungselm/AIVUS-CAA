from PIL import Image, ImageOps
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
    def resize_with_crop_or_pad_pil(img, target_height, target_width):
        """
        Resize an image with cropping or padding to maintain aspect ratio.

        Args:
            img: PIL Image object
            target_height: Target height in pixels
            target_width: Target width in pixels

        Returns:
            PIL Image object with target dimensions
        """
        # Get original dimensions
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        original_width, original_height = img.size
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height

        # Determine if we need to crop or pad
        if original_ratio > target_ratio:
            # Image is wider than target - crop horizontally or pad vertically
            # First resize to make height match target height
            new_height = target_height
            new_width = int(original_width * (new_height / original_height))
            img = img.resize((new_width, new_height), Image.BILINEAR)

            # Then crop or pad width
            if new_width > target_width:
                # Crop horizontally
                left = (new_width - target_width) // 2
                right = left + target_width
                img = img.crop((left, 0, right, target_height))
            else:
                # Pad horizontally
                padding = (target_width - new_width) // 2
                img = ImageOps.expand(img, (padding, 0, target_width - new_width - padding, 0), fill=0)
        else:
            # Image is taller than target - crop vertically or pad horizontally
            # First resize to make width match target width
            new_width = target_width
            new_height = int(original_height * (new_width / original_width))
            img = img.resize((new_width, new_height), Image.BILINEAR)

            # Then crop or pad height
            if new_height > target_height:
                # Crop vertically
                top = (new_height - target_height) // 2
                bottom = top + target_height
                img = img.crop((0, top, target_width, bottom))
            else:
                # Pad vertically
                padding = (target_height - new_height) // 2
                img = ImageOps.expand(img, (0, padding, 0, target_height - new_height - padding), fill=0)

        return img

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
