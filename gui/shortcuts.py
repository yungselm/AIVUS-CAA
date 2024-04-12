import time

from loguru import logger
from functools import partial
from PyQt5.QtGui import QKeySequence, QDesktopServices
from PyQt5.QtWidgets import QShortcut, QApplication
from PyQt5.QtCore import Qt, QUrl

from gui.error_message import ErrorMessage
from input_output.metadata import MetadataWindow
from input_output.read_image import read_image
from input_output.contours_io import write_contours
from report.report import report
from gui.display.contours_gui import new_contour, new_measure
from segmentation.segment import segment


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

def init_menu(main_window):
    file_menu = main_window.menu_bar.addMenu('File')
    open_action = file_menu.addAction('Open File', partial(read_image, main_window))
    open_action.setShortcut('Ctrl+O')
    file_menu.addSeparator()
    file_menu.addAction('Save Contours', partial(write_contours, main_window))
    file_menu.addAction('Save Report', partial(report, main_window))
    file_menu.addSeparator()
    exit_action = file_menu.addAction('Exit', main_window.close)
    exit_action.setShortcut('Ctrl+Q')

    edit_menu = main_window.menu_bar.addMenu('Edit')
    manual_contour = edit_menu.addAction('Manual Contour', partial(new_contour, main_window))
    manual_contour.setShortcut('E')
    edit_menu.addSeparator()
    measure_1 = edit_menu.addAction('Measurement 1', partial(new_measure, main_window, index=0))
    measure_1.setShortcut('1')
    measure_2 = edit_menu.addAction('Measurement 2', partial(new_measure, main_window, index=1))
    measure_2.setShortcut('2')

    view_menu = main_window.menu_bar.addMenu('View')
    hide_contours_action = view_menu.addAction('Hide Contours', partial(hide_contours, main_window))
    hide_contours_action.setShortcut('H')
    hide_special_points_action = view_menu.addAction('Hide Special Points', partial(hide_special_points, main_window))
    hide_special_points_action.setShortcut('G')
    view_menu.addSeparator()
    reset_windowing_action = view_menu.addAction('Reset Windowing', partial(reset_windowing, main_window))
    reset_windowing_action.setShortcut('R')
    toggle_color_action = view_menu.addAction('Toggle Color', partial(toggle_color, main_window))
    toggle_color_action.setShortcut('C')
    view_menu.addSeparator()
    filter_1 = view_menu.addAction('Apply Median Blur', partial(toggle_filter, main_window, index=0))
    filter_1.setShortcut('3')
    filter_2 = view_menu.addAction('Apply Gaussian Blur', partial(toggle_filter, main_window, index=1))
    filter_2.setShortcut('4')
    filter_3 = view_menu.addAction('Apply Bilateral Filter', partial(toggle_filter, main_window, index=2))
    filter_3.setShortcut('5')

    run_menu = main_window.menu_bar.addMenu('Run')
    run_menu.addAction('Extract Diastolic and Systolic Frames', main_window.contour_based_gating)
    run_menu.addAction('Automatic Segmentation', partial(segment, main_window))

    metadata_menu = main_window.menu_bar.addMenu('Metadata')
    metadata_menu.addAction('Show Metadata', partial(show_metadata, main_window))

    help_menu = main_window.menu_bar.addMenu('Help')
    help_menu.addAction('GitHub Page', partial(open_url, main_window, description='github'))
    help_menu.addAction('Keyboard Shortcuts', partial(open_url, main_window, description='keyboard_shortcuts'))
    help_menu.addSeparator()
    help_menu.addAction('About', partial(open_url, main_window))

def show_metadata(main_window):
    if main_window.image_displayed:
        metadata_window = MetadataWindow(main_window)
        metadata_window.show()

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
