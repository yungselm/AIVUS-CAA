import time

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QApplication
from PyQt5.QtCore import Qt

from input_output.read_image import read_image
from gui.display.contours_gui import new_contour


def init_shortcuts(main_window):
    # General
    QShortcut(QKeySequence('Ctrl+Q'), main_window, main_window.close)
    QShortcut(QKeySequence('Ctrl+O'), main_window, lambda: read_image(main_window))
    QShortcut(QKeySequence('H'), main_window, lambda: hide_contours(main_window))
    QShortcut(QKeySequence('J'), main_window, lambda: jiggle_frame(main_window))
    QShortcut(QKeySequence('E'), main_window, lambda: new_contour(main_window))
    # Windowing
    QShortcut(QKeySequence('R'), main_window, lambda: reset_windowing(main_window))
    QShortcut(QKeySequence('C'), main_window, lambda: toggle_color(main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, lambda: main_window.display_slider.next_gated_frame())
    QShortcut(QKeySequence(Qt.Key_Up), main_window, lambda: main_window.display_slider.next_gated_frame())
    QShortcut(QKeySequence('A'), main_window, lambda: main_window.display_slider.last_frame())
    QShortcut(QKeySequence(Qt.Key_Left), main_window, lambda: main_window.display_slider.last_frame())
    QShortcut(QKeySequence('S'), main_window, lambda: main_window.display_slider.last_gated_frame())
    QShortcut(QKeySequence(Qt.Key_Down), main_window, lambda: main_window.display_slider.last_gated_frame())
    QShortcut(QKeySequence('D'), main_window, lambda: main_window.display_slider.next_frame())
    QShortcut(QKeySequence(Qt.Key_Right), main_window, lambda: main_window.display_slider.next_frame())


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


def reset_windowing(main_window):
    main_window.display.window_level = main_window.display.initial_window_level
    main_window.display.window_width = main_window.display.initial_window_width
    main_window.display.display_image(update_image=True)


def toggle_color(main_window):
    main_window.colormap_enabled = not main_window.colormap_enabled
    main_window.display.display_image(update_image=True)
