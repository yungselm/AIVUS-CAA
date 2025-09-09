# docs/contents/key_features.rst

Key Features
============

The main functionalities of AIVUS-CAA are:

- **IVUS Image Inspection:** Frame-by-frame visualization of DICOM or NIfTI IVUS images, with display of associated metadata:contentReference[oaicite:5]{index=5}.
- **Manual Contouring:** Draw or adjust lumen contours on each frame, with automatic calculation of lumen area, circumference, and ellipticity.
- **Automatic Segmentation (WIP):** Experimental tools for automatic lumen segmentation across all frames.
- **Cardiac Gating:** Identify systolic and diastolic frames via signal processing, and tag them interactively.
- **Distance Measurements:** Measure up to two distances per frame (e.g. stent length) and include them in the analysis report.
- **Session Auto-save:** Automatic saving of contours and frame tags at user-defined intervals.
- **Reporting:** Generate a detailed report file summarizing measurements (areas, perimeters, ratios) for each frame.
- **Data Export:** Save images and contours to NIfTI or CSV for further analysis (e.g. ML training).

