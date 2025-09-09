# docs/contents/usage.rst

Usage
=====

After installation, the AIVUS-CAA GUI provides the following functionality:

- **Open Data:** Press `Ctrl+O` or use the menu to load an IVUS image series in DICOM or NIfTI format. 
- **Navigate Frames:** Use the `A` (previous) and `D` (next) keys to move frame-by-frame. If gating is enabled, use `W`/ `S` to move between diastolic and systolic frames.
- **Draw Contours:** Click on the IVUS image to add draggable control points on the lumen wall. Press `Enter` to finalize the contour. The lumen area and perimeter are updated in real-time.
- **Gating:** If enabled, the application will automatically segment the signal to identify cardiac cycles. The user can also manually tag frames as systolic or diastolic.
- **Measurements:** Use the tools to draw up to two distance lines per frame. These distances (e.g. stent length) are recorded in the report.
- **Saving:** Contours, tags, and measurements are auto-saved by default. You can also manually save the session via the `Save` button.
- **Export:** Save the current segmentation as a NIfTI file, or export measured values to CSV for further analysis.

Refer to the interactive help within the GUI for additional shortcuts and tips.