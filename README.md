## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Functionalities](#functionalities)
- [Configuration](#configuration)
- [Usage](#usage)
- [Keyboard shortcuts](#keyboard-shortcuts)


## Installation
```
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
```

If you plan on using GPU acceleration for model training, make sure to install the required tools (NVIDIA toolkit, etc.) and the corresponding version of Pytorch.

## Functionalities
This application is designed for IVUS images in DICOM format and offers the following functionalities:
- Inspect IVUS images frame-by-frame and display DICOM metadata
- Manually **draw lumen contours** with automatic calculation of lumen area, circumference and elliptic ratio
- Manually tag diastolic/systolic frames and frames with plaque
- **Auto-save** of contours and tags enabled by default with user-definable interval
- Generation of report file containing detailed metrics for each frame
- Ability to save images and segmentations as **NIfTI files**, e.g. to train a machine learning model
- (unfinished) Automatically extract diastolic/systolic frames
- (unfinished) Automatical segmentation of lumen for all frames

## Configuration
Make sure to quickly check the **config.yaml** file and configure everything to your needs.

## Usage
After the config file is set up properly, you can run the application using:
```
python3 main.py
```

## Keyboard shortcuts