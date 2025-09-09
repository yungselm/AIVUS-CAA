from os.path import join
BASE_DIR = "Dataset"
RESULT_PATH = "Results"
DATASET_NAME = "IVUS"

dataset_results = join(RESULT_PATH, DATASET_NAME)
dataset_raw = join(RESULT_PATH, DATASET_NAME)


ds_train_path = join(BASE_DIR, "imagesTr")
ds_train_seg_path = join(BASE_DIR, "labelsTr")
ds_test_path = join(BASE_DIR, "imagesTs")
ds_test_seg_path = join(BASE_DIR, "labelsTs")
ds_val_path = join(BASE_DIR, "imagesVal")
ds_val_seg_path = join(BASE_DIR, "labelsVal")



SHAPE = (512, 512, 1)
TRIAL_TYPE = 'MINMAX'
MODEL_NAME = "u2net_2d"
TRIAL_IDENTIFIER = f'{MODEL_NAME}_{TRIAL_TYPE}_{SHAPE[0]}'
LACT = "Sigmoid"
LR = 0.001
DECAY_STEP = 500
DECAY_RATE = 0.95
SEED = 1234
EPOCHS = 250

IMG_SIZE = (512, 512)
N_CHANNELS = 1
BATCH_SIZE = 16

BEST_SUFFIX = "_best"
LAST_SUFFIX = "_last"



VAL_SIZE = 0.1