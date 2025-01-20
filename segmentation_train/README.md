# IVUS Segmentation Training
Here we present the codes for training the segmentation models.

The setup should be like the following
```commandline
├── segmentation_train
│   ├──configs.py
│   ├──data_preprocessing.py
│   ├──ivus_3d_to_2d.py
│   ├──metrics.py
│   ├──models.py
│   ├──README.md
│   ├──requirements.txt
│   ├──train.py
│   ├──Dataset
│   │  ├──imagesTr
│   │  │   ├── img_01.nii.gz
│   │  │   ├── ...
│   │  ├──labelsTr
```

# Installation
## Python Version
```commandline
Python 3.10.16
```

Installing all the modules
```commandline
pip install -r requirements.txt
```

# Data Preparation
In case of dicom images you can convert it to nifti files using:
```commandline
python ivus_3d_to_3d.py --input_data path-to-dicom-directory --output output_nifti
```

Then copy the images and segmentation to `Dataset/imagesTr` and `Dataset/labelsTr` respectively.

# Training
The hyper parameters are located in the `configs.py`. After modifying them, run the following:

```commandline
python train.py
```

Two models will be generated at the end, `_last.h5` and `_best.h5` inside `SAVE_DIR` directory. 

# Inference
Use the `predict.py` code to predict on your test dataset.
```commandline
python predict.py --input_data path-to-non-labeled-data --output_data path-to-predicted-masks
```

