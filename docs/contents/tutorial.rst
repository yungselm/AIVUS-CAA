.. docs/contents/tutorial.rst

Tutorial
========

A typical workflow in AIVUS-CAA involves the following steps:

Segmentation
------------

1. Open the IVUS data file via **File → Open** or with :kbd:`Ctrl+O`.
2. Click on `Automatic Segmentation` to let the software pre-segment the lumen in all frames (optional). Different ML models can be specified in the config.yaml file.

.. image:: ../../media/explanation_software_part1.gif
   :alt: Example figure
   :align: center
   :width: 800px

3. Hold the right mouse button :kbd:`RMB` for windowing (can be reset by pressing :kbd:`R`)
4. Use :kbd:`A` and :kbd:`D` to navigate through frames (or use the slider below the image).
5. New contours can be drawn by using :kbd:`E` and then draw a new contour by left clicking. To finalize the contour click on the initially set point. The points on the contour can be manipulated by just dragging and new points can be set by clicking on the contour.
6. :kbd:`Ctrl+Z` to undo the last action.

.. image:: ../../media/explanation_software_part2.gif
   :alt: Example figure
   :align: center
   :width: 800px

Gating
------
Gating can only be performed as soon as a lumen contour is available for each frame.

7. Click on the `Gating` button to let the software compute a gating signal based on image and contour properties. An automatic estimate for systolic and diastolic phases will be set based on overlapping peaks and ellipticity of the overall frames. 
You can choose the range to gate in by providing frame range and additionally define if peaks should be maxima or extrema for curves (typically maxima for image-based and extrema for contour-based curves).
8. If the results are not satisfying they can be deleted at once with :kbd:`Alt+Del`
8. New phases can be set by clicking in the gating window in the top right. Existing phases can be moved by dragging.
9. Tag desired frames as systolic (**S**) or diastolic (**D**).
10. View results with :kbd:`Alt+P` and export as needed.

.. image:: ../../media/explanation_software_part3.gif
   :alt: Example figure
   :align: center
   :width: 800px