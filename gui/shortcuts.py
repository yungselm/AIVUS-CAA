from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from input_output.read_image import read_image


def init_shortcuts(main_window):
        QShortcut(QKeySequence('Ctrl+Q'), main_window, main_window.close)
        QShortcut(QKeySequence('Ctrl+O'), main_window, lambda _: read_image(main_window))
        QShortcut(QKeySequence('H'), main_window, lambda _: hide_contours(main_window))

def hide_contours(main_window):
    if main_window.image_displayed:
        if not main_window.hide_contours_box.isChecked():
                main_window.hide_contours_box.setChecked(True)
        elif main_window.hide_contours_box.isChecked():
                main_window.hide_contours_box.setChecked(False)
        main_window.hide_contours_box.setChecked(main_window.hide_contours_box.isChecked())
