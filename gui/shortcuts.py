import time

from functools import partial
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QApplication, QMessageBox
from PyQt5.QtCore import Qt

from gui.display.contours_gui import new_contour, new_measure


def init_shortcuts(main_window):
    # General
    QShortcut(QKeySequence('H'), main_window, partial(hide_contours, main_window))
    QShortcut(QKeySequence('J'), main_window, partial(jiggle_frame, main_window))
    QShortcut(QKeySequence('E'), main_window, partial(new_contour, main_window))
    QShortcut(QKeySequence('1'), main_window, partial(new_measure, main_window, index=0))
    QShortcut(QKeySequence('2'), main_window, partial(new_measure, main_window, index=1))
    QShortcut(QKeySequence('Escape'), main_window, partial(stop_all, main_window))
    QShortcut(QKeySequence('Delete'), main_window, partial(delete_contour, main_window))
    QShortcut(QKeySequence('Ctrl+Z'), main_window, partial(undo_delete, main_window))
    # Windowing
    QShortcut(QKeySequence('R'), main_window, partial(reset_windowing, main_window))
    QShortcut(QKeySequence('C'), main_window, partial(toggle_color, main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Up), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence('A'), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence(Qt.Key_Left), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence('S'), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Down), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence('D'), main_window, main_window.display_slider.next_frame)
    QShortcut(QKeySequence(Qt.Key_Right), main_window, main_window.display_slider.next_frame)


def display_shortcuts_info(main_window):
    text = (
        '\n'
        'First, load a DICOM/NIfTi file using the button below or by pressing Ctrl+O.\n'
        'If available, contours for that file will be read automatically.\n'
        'Use the A and D keys to move through all frames, S and W keys to move through gated frames.\n'
        'Press E to draw a new Lumen contour.\n'
        'Press Delete to delete the current Lumen contour.\n'
        'Hold the right mouse button for windowing (can be reset by pressing R).\n'
        'Press C to toggle color mode.\n'
        'Press H to hide all contours.\n'
        'Press J to jiggle around the current frame.\n'
        'Press Ctrl+Q to close the program.\n'
    )
    QMessageBox.information(main_window, 'Keyboard Shortcuts', text)


def hide_contours(main_window):
    if main_window.image_displayed:
        if not main_window.hide_contours_box.isChecked():
            main_window.hide_contours_box.setChecked(True)
        elif main_window.hide_contours_box.isChecked():
            main_window.hide_contours_box.setChecked(False)


def jiggle_frame(main_window):
    if main_window.image_displayed:
        current_frame = main_window.display_slider.value()
        main_window.display_slider.setValue(current_frame + 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.setValue(current_frame)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.setValue(current_frame - 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.setValue(current_frame)
        QApplication.processEvents()


def stop_all(main_window):
    main_window.display.stop_contour()
    main_window.display.measure_index = None


def delete_contour(main_window):
    if main_window.image_displayed:
        main_window.tmp_lumen_x = main_window.data['lumen'][0][main_window.display.frame]  # for Ctrl+Z
        main_window.tmp_lumen_y = main_window.data['lumen'][1][main_window.display.frame]
        main_window.data['lumen'][0][main_window.display.frame] = []
        main_window.data['lumen'][1][main_window.display.frame] = []
        main_window.display.display_image(update_contours=True)


def undo_delete(main_window):
    if main_window.image_displayed and main_window.tmp_lumen_x:
        main_window.data['lumen'][0][main_window.display.frame] = main_window.tmp_lumen_x
        main_window.data['lumen'][1][main_window.display.frame] = main_window.tmp_lumen_y
        main_window.tmp_lumen_x = []
        main_window.tmp_lumen_y = []
    main_window.display.stop_contour()


def reset_windowing(main_window):
    if main_window.image_displayed:
        main_window.display.window_level = main_window.display.initial_window_level
        main_window.display.window_width = main_window.display.initial_window_width
        main_window.display.display_image(update_image=True)


def toggle_color(main_window):
    if main_window.image_displayed:
        main_window.colormap_enabled = not main_window.colormap_enabled
        main_window.display.display_image(update_image=True)
