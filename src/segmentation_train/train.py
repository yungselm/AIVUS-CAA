import tensorflow as tf

from deep_utils import DirUtils
from configs import *
from data_preprocessing import DataGenerator
from metrics import dice_score_tf
from models import get_model
from utils import split_data

SAVE_DIR = f"models"
SAVE_PATH = f"{DirUtils.mkdir_incremental(SAVE_DIR)}/{TRIAL_IDENTIFIER}.h5"

if __name__ == '__main__':
    model, model_name = get_model(MODEL_NAME)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=DECAY_STEP,
        decay_rate=DECAY_RATE)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[dice_score_tf])
    my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=DirUtils.split_extension(SAVE_PATH, suffix="_best"),
                                                       save_best_only=True),
                    tf.keras.callbacks.TensorBoard(log_dir=SAVE_PATH)]
    train_img_files, train_mask_files = DirUtils.list_dir_full_path(ds_train_path), DirUtils.list_dir_full_path(ds_train_seg_path)
    val_img_files, val_mask_files = DirUtils.list_dir_full_path(ds_val_path), DirUtils.list_dir_full_path(ds_val_seg_path)

    print(f"[INFO] Loading Datasets")
    val_dataset = DataGenerator(val_img_files, val_mask_files, BATCH_SIZE, IMG_SIZE, N_CHANNELS, augmentation_p=0)
    train_dataset = DataGenerator(train_img_files, train_mask_files, BATCH_SIZE, IMG_SIZE, N_CHANNELS,
                                  augmentation_p=0.6)
    print(f"[INFO] Training on {len(train_img_files)} and validating on {len(val_img_files)}")
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        # verbose=1,
                        callbacks=my_callbacks
                        )

    model.save(DirUtils.split_extension(SAVE_PATH, suffix="_last"))

    print("[INFO] Evaluating the model: ")
    test_img_files, test_mask_files = DirUtils.list_dir_full_path(ds_test_path), DirUtils.list_dir_full_path(
        ds_test_seg_path)
    test_dataset = DataGenerator(test_img_files, test_mask_files, BATCH_SIZE, IMG_SIZE, N_CHANNELS, augmentation_p=0)
    model.evaluate(test_dataset)

