Architecture Overview
=====================

AIVUS-OCT is a desktop application built with PyQt6. The source code lives in ``src/`` and is organised into five packages around clear responsibilities.

Entry Point
-----------

``main.py`` initialises logging, applies the dark theme, and launches the ``Master`` window via Hydra configuration.

GUI (``gui/``)
--------------

The ``Master`` class in ``gui.py`` is the central coordinator. It owns all application state — loaded images, contours, gating data, metadata — and wires together every other module. Sub-packages handle distinct regions of the interface:

- **left_half/** — frame-by-frame image display and manual contouring canvas
- **right_half/** — longitudinal view and gating signal display
- **popup_windows/** — results plots, small previews, video player, and dialogs
- **utils/** — shared GUI helpers: geometry math, contour rendering, metrics, custom slider widget

Input / Output (``input_output/``)
-----------------------------------

- ``read_image.py`` — loads DICOM and NIfTI files, normalises pixel data, and populates the display
- ``metadata.py`` — parses DICOM tags (patient info, imaging parameters) and renders them in a table; handles IVUS and OCT modality differences
- ``contours_io.py`` — defines ``Contour``, ``FrameData``, and ``Measurements`` data structures; serialises and deserialises contour sessions to disk

Gating (``gating/``)
---------------------

Cardiac gating extracts diastolic/systolic frames from an image sequence.

- ``automatic_gating.py`` — dialog for selecting the gating method (maxima or extrema)
- ``contour_based_gating.py`` — derives a gating signal from contour area measurements over time
- ``signal_processing.py`` — shared signal processing utilities used by both gating methods

Segmentation (``segmentation/``)
----------------------------------

- ``segment.py`` — runs the neural network predictor on IVUS frames and converts output masks to contours with measurements
- ``predict.py`` — model loading and inference
- ``segment_files.py`` — batch segmentation of files outside the GUI
- ``save_as_nifti.py`` — exports segmentation masks as NIfTI volumes

Reporting (``report/``)
------------------------

``report.py`` aggregates per-frame lumen measurements into a summary report, generates result plots, and exports data as CSV files.
