from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QMenuBar,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QTableWidget,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer

from gui.main_window.left_half import LeftHalf
from gui.main_window.right_half import RightHalf
from gui.shortcuts import init_shortcuts, init_menu
from input_output.contours_io import write_contours
from gating.contour_based_gating import ContourBasedGating
from segmentation.predict import Predict
from segmentation.save_as_nifti import save_as_nifti


class Master(QMainWindow):
    """Main Window Class

    Attributes:
        image: bool, indicates whether images have been loaded (true) or not
        contours: bool, indicates whether contours have been loaded (true) or not
        segmentation: bool, indicates whether segmentation has been performed (true) or not
        lumen: tuple, contours for lumen border
        plaque: tuple, contours for plaque border
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.autosave_interval = config.save.autosave_interval
        self.use_xml_files = config.save.use_xml_files
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

    def closeEvent(self, event):
        """Tasks to be performed before actually closing the program"""
        if self.image_displayed:
            self.save_before_close()

    def save_before_close(self):
        """Save contours, etc before closing program or reading new DICOM file"""
        self.status_bar.showMessage('Saving contours and NIfTi files...')
        write_contours(self)
        save_as_nifti(self)
        self.status_bar.showMessage(self.waiting_status)

    def auto_save(self):
        """Automatically saves contours to a temporary file every autoSaveInterval seconds"""
        if self.image_displayed:
            write_contours(self)

    def errorMessage(self):
        """Helper function for errors"""

        warning = QMessageBox(self)
        warning.setWindowModality(Qt.WindowModal)
        warning.setWindowTitle('Error')
        warning.setText('Segmentation must be performed first')
        warning.exec_()

    def successMessage(self, task):
        """Helper function for success messages"""

        success = QMessageBox(self)
        success.setWindowModality(Qt.WindowModal)
        success.setWindowTitle('Status')
        success.setText(task + ' has been successfully completed')
        success.exec_()
