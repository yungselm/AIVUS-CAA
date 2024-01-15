from gui.error_message import ErrorMessage


def new_contour(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading DICOM file')
        return

    main_window.tmp_lumen_x = main_window.data['lumen'][0][main_window.display.frame]  # for Ctrl+Z
    main_window.tmp_lumen_y = main_window.data['lumen'][1][main_window.display.frame]

    main_window.display.start_drawing()
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True
