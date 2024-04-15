from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QMenuBar,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QStatusBar,
)
from PyQt5.QtCore import QTimer

from gui.left_half.left_half import LeftHalf
from gui.right_half.right_half import RightHalf
from gui.shortcuts import init_shortcuts, init_menu
from input_output.contours_io import write_contours
from gating.contour_based_gating import ContourBasedGating
from segmentation.predict import Predict
from segmentation.save_as_nifti import save_as_nifti


class Master(QMainWindow):
    """Main Window Class"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.autosave_interval = config.save.autosave_interval
        self.contour_based_gating = ContourBasedGating(self)
        self.predictor = Predict(self)
        self.image_displayed = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.filter = None
        self.tmp_lumen_x = []  # for Ctrl+Z
        self.tmp_lumen_y = []
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}  # container to be saved in JSON file later, includes contours, etc.
        self.metadata = {}  # metadata used outside of read_image (not saved to JSON file)
        self.images = None
        self.diastole_color = (39, 69, 219)
        self.systole_color = (209, 55, 38)
        self.measure_colors = ['red', 'cyan']
        self.waiting_status = 'Waiting for user input...'
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self):
        SPACING = 5

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        init_menu(self)
        self.metadata_table = QTableWidget()

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        central_widget = QWidget()
        main_window_hbox = QHBoxLayout()
        self.left_vbox = QVBoxLayout()
        self.left_vbox.setContentsMargins(0, 0, SPACING, SPACING)
        LeftHalf(self)
        main_window_hbox.addLayout(self.left_vbox)
        self.right_vbox = QVBoxLayout()
        self.right_vbox.setContentsMargins(SPACING, 0, 0, SPACING)
        RightHalf(self)
        main_window_hbox.addLayout(self.right_vbox)
        central_widget.setLayout(main_window_hbox)

        self.setWindowTitle('AAOCA Segmentation Tool')
        self.setCentralWidget(central_widget)
        self.showMaximized()

        timer = QTimer(self)
        timer.timeout.connect(self.auto_save)
        timer.start(self.autosave_interval)  # autosave interval in milliseconds

    def auto_save(self):
        if self.image_displayed:
            write_contours(self)

    def closeEvent(self, event):
        """Tasks to be performed before closing the program"""
        if self.image_displayed:
            self.status_bar.showMessage('Saving contours and NIfTi files...')
            write_contours(self)
            save_as_nifti(self)
            self.status_bar.showMessage(self.waiting_status)
