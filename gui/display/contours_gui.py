from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtCore import Qt


def new_contour(main_window):
    if not main_window.image_displayed:
        warning = QErrorMessage(main_window)
        warning.setWindowModality(Qt.WindowModal)
        warning.showMessage('Cannot create manual contour before reading DICOM file')
        warning.exec_()
        return

    main_window.tmp_lumen_x = main_window.data['lumen'][0][main_window.display.frame]  # for Ctrl+Z
    main_window.tmp_lumen_y = main_window.data['lumen'][1][main_window.display.frame]

    main_window.display.start_drawing()
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True
