Architecture Overview
=====================

AIVUS-OCT is a desktop application built with PyQt6. The source code lives in ``src/`` and is organised into packages around clear responsibilities::

    src/
    ├── domain/              data classes — RuntimeData, FrameData, CctaRuntimeData
    ├── gating/              cardiac gating — ContourBasedGating, AutomaticGating, signal processing
    ├── gui/                 app-level wiring — Master window, keyboard shortcuts, menu bar
    ├── input_output/
    │   ├── input/           readers — DICOM series, NIfTI (IVUS/OCT/CCTA), metadata parsers
    │   └── output/          writers — report CSV, NIfTI export, JSON contour save/load
    ├── pages/
    │   ├── ccta/            CCTA page — tri-plane viewer, VTK 3D renderer, mask panel
    │   └── intravascular/
    │       ├── left_half/   image display, spline contour editor, drawing tools
    │       ├── popup_windows/ dialogs — frame range, message boxes, settings
    │       ├── right_half/  gating plot, longitudinal view, phase controls
    │       └── utils/       helpers shared across the intravascular page
    ├── segmentation/        automatic segmentation — nnUZoo wrapper, mask-to-contour conversion
    └── tools/               shared Qt-independent tools — BrushGeometry, BrushCursor

Entry Point
-----------

``main.py`` initialises logging, applies the dark theme, and launches the ``Master`` window via Hydra configuration.

Domain (``domain/``)
---------------------

The single source of truth for runtime state. All loaded images, contours, gating data, and metadata are stored in ``RuntimeData`` (intravascular) and ``CctaRuntimeData`` (CCTA). Pages read and write through these dataclasses rather than caching data locally, which keeps page teardown and reset trivial.

GUI (``gui/``)
--------------

The ``Master`` class is the central coordinator. It owns the top-level window and wires together every page, the menu bar, and the keyboard shortcut table. It instantiates pages via ``reload_intravascular`` / ``reload_ccta``; tearing down a page and reinstantiating it is the reset strategy.

Input / Output (``input_output/``)
-----------------------------------

Has no GUI imports and can be exercised headlessly in tests or CLI scripts.

- **input/** — loads DICOM series and NIfTI volumes (IVUS, OCT, CCTA), normalises pixel data, parses DICOM tags (patient info, imaging parameters), and handles IVUS / OCT modality differences
- **output/** — exports report CSVs, NIfTI segmentation masks, and serialises / deserialises contour sessions (``FrameData``, ``Measurements``) to and from JSON on disk

Pages (``pages/``)
-------------------

All page-specific UI code lives here. Each page is a self-contained ``QWidget``.

- **ccta/** — tri-plane MPR viewer, VTK-based 3D renderer, and mask editing panel for CCTA data
- **intravascular/** — the IVUS / OCT review page, further divided into:

  - *left_half/* — frame-by-frame image display and spline-based contour editor with manual drawing tools
  - *right_half/* — longitudinal cross-section view, gating signal plot, and cardiac phase controls
  - *popup_windows/* — frame-range dialogs, results plots, small previews, video player, and message boxes
  - *utils/* — helpers shared across the intravascular page: geometry math, contour rendering, area metrics, and the custom slider widget

Gating (``gating/``)
---------------------

Cardiac gating extracts diastolic / systolic frames from an image sequence.

- ``automatic_gating.py`` — dialog for selecting the gating method (maxima or extrema)
- ``contour_based_gating.py`` — derives a gating signal from contour area measurements over time
- ``signal_processing.py`` — shared signal-processing utilities used by both gating methods

Segmentation (``segmentation/``)
----------------------------------

- ``segment.py`` — runs the neural network predictor on IVUS / OCT frames and converts output masks to contours with measurements
- ``predict.py`` — model loading and inference (nnUZoo wrapper)
- ``segment_files.py`` — batch segmentation of files outside the GUI
- ``save_as_nifti.py`` — exports segmentation masks as NIfTI volumes

Tools (``tools/``)
-------------------

Qt-independent utilities reusable across pages: pure geometry helpers (``BrushGeometry``) and pixmap cursor construction (``BrushCursor``). No widget imports — safe to unit-test without a display.
