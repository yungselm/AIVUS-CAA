import tensorflow as tf
from sklearn.model_selection import train_test_split

from deep_utils import DirUtils
from configs import *
from data_preprocessing import DataGenerator
from metrics import dice_score_tf
from models import get_model

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
    img_files = DirUtils.list_dir_full_path(ds_train_path)
    mask_files = DirUtils.list_dir_full_path(ds_train_seg_path)
    train_img_files, val_img_files, train_mask_files, val_mask_files = train_test_split(img_files,
                                                                                        mask_files,
                                                                                        test_size=VAL_SIZE,
                                                                                        random_state=SEED)
    train_dataset = DataGenerator(train_img_files, train_mask_files, BATCH_SIZE, IMG_SIZE, N_CHANNELS,
                                  augmentation_p=0.6)
    val_dataset = DataGenerator(val_img_files, val_mask_files, BATCH_SIZE, IMG_SIZE, N_CHANNELS, augmentation_p=0)

    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        # verbose=1,
                        callbacks=my_callbacks
                        )

    model.save(DirUtils.split_extension(SAVE_PATH, suffix="_last"))
