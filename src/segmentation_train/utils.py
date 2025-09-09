from sklearn.model_selection import train_test_split

from deep_utils import DirUtils

def split_data(ds_train_path:str, ds_train_seg_path:str, test_size: float, random_state: int):
    """
    Patient Wise split. If the filenames follow the following structure:
    patientname_other_info.nii.gz
    :param ds_train_path:
    :param ds_train_seg_path:
    :param test_size:
    :param random_state:
    :return:
    """
    img_files = DirUtils.list_dir_full_path(ds_train_path, interest_extensions=".gz", return_dict=True)
    mask_files = DirUtils.list_dir_full_path(ds_train_seg_path, interest_extensions=".gz", return_dict=True)

    patient_names = {name.split("_")[0] for name in img_files.keys()}
    train_patients, test_patients = train_test_split(list(patient_names), test_size=test_size, random_state=random_state)

    train_images = [filepath for filename, filepath in img_files.items() if filename.split("_")[0] in train_patients]
    train_labels = [filepath for filename, filepath in mask_files.items() if filename.split("_")[0] in train_patients]

    test_images = [filepath for filename, filepath in img_files.items() if filename.split("_")[0] in test_patients]
    test_labels = [filepath for filename, filepath in mask_files.items() if filename.split("_")[0] in test_patients]
    return train_images, train_labels, test_images, test_labels

