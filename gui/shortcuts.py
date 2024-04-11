import time

from functools import partial
from PyQt5.QtGui import QKeySequence, QDesktopServices
from PyQt5.QtWidgets import QShortcut, QApplication
from PyQt5.QtCore import Qt, QUrl

from gui.error_message import ErrorMessage


def init_shortcuts(main_window):
    # General
    QShortcut(QKeySequence('J'), main_window, partial(jiggle_frame, main_window))
    QShortcut(QKeySequence('Escape'), main_window, partial(stop_all, main_window))
    QShortcut(QKeySequence('Delete'), main_window, partial(delete_contour, main_window))
    QShortcut(QKeySequence('Ctrl+Z'), main_window, partial(undo_delete, main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Up), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence('A'), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence(Qt.Key_Left), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence('S'), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Down), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence('D'), main_window, main_window.display_slider.next_frame)
    QShortcut(QKeySequence(Qt.Key_Right), main_window, main_window.display_slider.next_frame)


def open_url(main_window, description=None):
    if description == 'github':
        url = 'https://github.com/cardionaut/AAOCASeg'
    elif description == 'keyboard_shortcuts':
        url = 'https://github.com/cardionaut/AAOCASeg?tab=readme-ov-file#keyboard-shortcuts'
    else:
        url = 'https://www.youtube.com/watch?v=xvFZjo5PgG0'
    if not QDesktopServices.openUrl(QUrl(url)):
        ErrorMessage(main_window, 'Could not open the browser. Please visit\n' + url)


def hide_contours(main_window):
    if main_window.image_displayed:
        main_window.hide_contours_box.setChecked(not main_window.hide_contours_box.isChecked())


def hide_special_points(main_window):
    if main_window.image_displayed:
        main_window.hide_special_points_box.setChecked(not main_window.hide_special_points_box.isChecked())


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


def toggle_filter(main_window, index):
    if main_window.image_displayed:
        if main_window.filter == index:
            main_window.filter = None
        else:
            main_window.filter = index
        main_window.display.display_image(update_image=True)


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
