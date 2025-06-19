
# NIVA (Neo-IntraVascular AAOCA assessment) <!-- omit in toc -->

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
  - [Basic](#basic)
  - [Creating an executable file](#creating-an-executable-file)
- [Functionalities](#functionalities)
- [Configuration](#configuration)
- [Usage](#usage)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Acknowledgements](#acknowledgements)

## Installation

### Basic

```bash
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
```

If you plan on using GPU acceleration for model training and inference, make sure to install the required tools (NVIDIA toolkit, etc.) and the corresponding version of Tensorflow.

The program was tested on Ubuntu 22.04.5 with python 3.10.12. We tested it on differnt hardware, NVIDIA drivers and CUDA tended to cause problems cross-platforms. Make sure to download the corresponding drivers and CUDA toolkit, e.g.:
```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential dkms
sudo ubuntu-drivers autoinstall
sudo reboot
# verify the installation of the driver
nvidia-smi
sudo apt install nvidia-cuda-toolkit
```
Potentially extra steps are needed.

### Creating an executable file (**optional**)

In case you prefer a single executable file instead of running the application from the terminal or an IDE, you can follow these steps to generate an executable for **the OS you are executing the command on**, i.e. if you run this in a Linux environment, you will create an executable for Linux.\
First, you need to install the application on your system (e.g. with the commands above on Linux) and ensure it is properly set up and can be run.\
Then you can use Pyinstaller to create an executable:

```bash
pyinstaller -F --hiddenimport=pydicom.encoders.gdcm --hiddenimport=pydicom.encoders.pylibjpeg --hiddenimport=scipy.special._cdflib main.py
```

You can find the executable **main** in the **dist** folder.
If you are having trouble launching the application from the executable, try running it in a terminal to see potential errors:

```bash
./dist/main
```

## Functionalities

This application is designed for IVUS images in DICOM or NIfTi format and offers the following functionalities:

- Inspect IVUS images frame-by-frame and display DICOM metadata
- Manually **draw lumen contours** with automatic calculation of lumen area, circumference and elliptic ratio
- **Automatic segmentation** of lumen for all frames (work in progress)
- **Automatic gating** with extraction of diastolic/systolic frames
- Manually tag diastolic/systolic frames
- Ability to measure up to two distances per frame which will be stored in the report
- **Auto-save** of contours and tags enabled by default with user-definable interval
- Generation of report file containing detailed metrics for each frame
- Ability to save images and segmentations as **NIfTi files**, e.g. to train a machine learning model

## Configuration

Make sure to quickly check the **config.yaml** file and configure everything to your needs.

## Usage

After the config file is set up properly, you can run the application using:

```bash
python3 src/main.py
```

This will open a graphical user interface (GUI) in which you have access to the above-mentioned functionalities.

## Keyboard shortcuts

For ease-of-use, this application contains several keyboard shortcuts.\
In the current state, these cannot be changed by the user (at least not without changing the source code).

- Press <kbd>Ctrl</kbd> + <kbd>O</kbd> to open a DICOM/NIfTi file
- Use the <kbd>A</kbd> and <kbd>D</kbd> keys to move through the IVUS images frame-by-frame
- If gated (diastolic/systolic) frames are available, you can move through those using <kbd>S</kbd> and <kbd>W</kbd>\
  Make sure to select which gated frames you want to traverse using the corresponding button (blue for diastolic, red for systolic)
- Press <kbd>E</kbd> to manually draw a new lumen contour\
  In case you accidentally delete a contour, you can use <kbd>Ctrl</kbd> + <kbd>Z</kbd> to undo
- Use <kbd>1</kbd>, <kbd>2</kbd> to draw measurements 1 and 2, respectively
- Use <kbd>3</kbd>, <kbd>4</kbd> or <kbd>5</kbd> to apply image filters
- Hold the right mouse button <kbd>RMB</kbd> for windowing (can be reset by pressing <kbd>R</kbd>)
- Press <kbd>C</kbd> to toggle color mode
- Press <kbd>H</kbd> to hide all contours
- Press <kbd>J</kbd> to jiggle around the current frame
- Press <kbd>Ctrl</kbd> + <kbd>S</kbd> to manually save contours (auto-save is enabled by default)
- Press <kbd>Ctrl</kbd> + <kbd>R</kbd> to generate report file
- Press <kbd>Ctrl</kbd> + <kbd>Q</kbd> to close the program
- Press <kbd>Alt</kbd> + <kbd>P</kbd> to plot the results for gated frames (difference area systole and diastole, by distance)
- Press <kbd>Alt</kbd> + <kbd>Delete</kbd> to define a range of frames to remove gating
- Press <kbd>Alt</kbd> + <kbd>S</kbd> to define a range of frames to switch systole and diastole in gated frames

## Tutorial
### Window manipulation:
![Demo](media/explanation_software_part1.gif)
### Contour manipulation:
![Demo](media/explanation_software_part2.gif)
### Gating module:
This module implements gating by analyzing both image-derived metrics (e.g., pixel-wise correlation and blurriness) and vector-based contour measurements (e.g., distance and direction from the image center to each contour centroid). Changes in these metrics are displayed over the sequence of frames during a pullback.

The resting phases of the cardiac cycle—diastole and systole—are characterized by minimal vessel motion for several consecutive frames. We visualize these phases using two curves: the image-based curve (green) represents metrics such as correlation peaks and minimal blurriness, while the contour-based curve (yellow) reflects extrema in the vector measurements (i.e., alternating peaks and valleys corresponding to systolic and diastolic positions).

- **Image-Based Metrics**: Select local maxima corresponding to frames with the highest pixel correlation and lowest blurriness.

- **Contour-Based Metrics**: Select extrema in the distance vector, capturing the transition between diastole and systole.

Movement patterns may vary between datasets; consequently, the final frame selection is left to the user.

**Peak Assignment**: Detected peaks in each curve are matched by intersecting their frame indices. We apply a Butterworth filter (passband: 45–180 bpm) to smooth each curve; the unfiltered signal is displayed as a dotted line beneath the filtered curve.

**Interactive Gating Interface**:
- Range Selection: Specify the frame interval for gating.
- Zoom & Pan: Zoom into the plot and drag lines to adjust gating thresholds or remove unwanted markers by dragging them downward.
- Compare Frames: Click "Compare Frames" to open the nearest proximal frame for the selected phase (systole or diastole).

![Demo](media/explanation_software_part3.gif)

## Acknowledgements

The starting point of this project was the [DeepIVUS](https://github.com/dmolony3/DeepIVUS) public repository by [David](https://github.com/dmolony3) (cloned on May 22, 2023).


# Citation
Please kindly cite the following paper if you use this repository.

```
@article{Stark2025,
  author = {Anselm W. Stark and Pooya Mohammadi and Sebastian Balzer and Marc Ilic and Manuel Bergamin
            and Ryota Kakizaki and Andreas Giannopoulos and Andreas Haeberlin and Lorenz Räber
            and Isaac Shiri and Christoph Gräni},
  title = {Automated IntraVascular UltraSound Image Processing and Quantification
           of Coronary Artery Anomalies: The AIVUS-CAA software},
  journal = {medRxiv},
  publisher = {Cold Spring Harbor Laboratory Press},
  year = {2025},
  doi       = {10.1101/2025.02.18.25322450},
  url       = {http://medrxiv.org/content/early/2025/02/20/2025.02.18.25322450.abstract},
  note      = {Preprint}
}
```
```
Stark, A. W., P. Mohammadi Kazaj, S. Balzer, M. Ilic, M. Bergamin, R. Kakizaki,
A. Giannopoulos, A. Haeberlin, L. Raber, I. Shiri and C. Grani (2025).
"Automated IntraVascular UltraSound Image Processing and Quantification of
Coronary Artery Anomalies: The AIVUS-CAA software." medRxiv: 2025.2002.2018.25322450.

```
