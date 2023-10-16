import time
import bisect

from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHeaderView,
    QStyle,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QErrorMessage,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

from gui.display import Display
from gui.slider import Slider, Communicate
from input_output.read_dicom import readDICOM
from input_output.contours import writeContours, segment, newSpline
from input_output.report import report
from preprocessing.preprocessing import PreProcessing
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
        self.use_xml_files = False
        self.image_displayed = False
        self.contours_drawn = False
        self.segmentation = False
        self.colormap_enabled = False
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {}  # container to be saved in JSON file later, includes contours, etc.
        self.metadata = {}  # metadata used outside of readDICOM (not saved to JSON file)
        self.images = None
        self.initGUI()

    def initGUI(self):
        spacing = 5
        self.setGeometry(spacing, spacing, 1200, 1200)
        self.display_size = 800
        self.addToolBar("MY Window")
        self.showMaximized()

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Waiting for user input...')
        layout = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox1hbox1 = QHBoxLayout()

        vbox1.setContentsMargins(0, 0, spacing, spacing)
        vbox2.setContentsMargins(spacing, 0, 0, spacing)
        vbox2hbox1 = QHBoxLayout()
        vbox2hbox2 = QHBoxLayout()
        vbox2.addLayout(vbox2hbox1)
        vbox2.addLayout(vbox2hbox2)
        layout.addLayout(vbox1)
        layout.addLayout(vbox2)

        dicomButton = QPushButton('Read DICOM')
        gatingButton = QPushButton('Extract Diastolic and Systolic Frames')
        segmentButton = QPushButton('Segment')
        splineButton = QPushButton('Manual Contour')
        writeButton = QPushButton('Write Contours')
        reportButton = QPushButton('Write Report')

        dicomButton.setToolTip("Load images in .dcm format")
        gatingButton.setToolTip("Extract diastolic and systolic images from pullback")
        segmentButton.setToolTip("Run deep learning based segmentation of lumen")
        splineButton.setToolTip("Manually draw new contour for lumen")
        writeButton.setToolTip("Save contours in .xml file")
        reportButton.setToolTip("Write report to .txt file")

        hideHeader1 = QHeaderView(Qt.Vertical)
        hideHeader1.hide()
        hideHeader2 = QHeaderView(Qt.Horizontal)
        hideHeader2.hide()
        self.infoTable = QTableWidget()
        self.infoTable.setRowCount(8)
        self.infoTable.setColumnCount(2)
        self.infoTable.setItem(0, 0, QTableWidgetItem('Patient Name'))
        self.infoTable.setItem(1, 0, QTableWidgetItem('Patient DOB'))
        self.infoTable.setItem(2, 0, QTableWidgetItem('Patient Sex'))
        self.infoTable.setItem(3, 0, QTableWidgetItem('Pullback Speed'))
        self.infoTable.setItem(4, 0, QTableWidgetItem('Resolution (mm)'))
        self.infoTable.setItem(5, 0, QTableWidgetItem('Dimensions'))
        self.infoTable.setItem(6, 0, QTableWidgetItem('Manufacturer'))
        self.infoTable.setItem(7, 0, QTableWidgetItem('Model'))
        self.infoTable.setVerticalHeader(hideHeader1)
        self.infoTable.setHorizontalHeader(hideHeader2)
        self.infoTable.horizontalHeader().setStretchLastSection(True)

        autoSaveInterval = 10000  # milliseconds
        self.shortcutInfo = QLabel()
        self.shortcutInfo.setText(
            (
                '\n'
                'First, load a DICOM file using the button below.\n'
                'If available, contours for that file will be read automatically.\n'
                'Use the A and D keys to move through all frames, S and W keys to move through gated frames.\n'
                'Press E to draw a new Lumen contour.\n'
                'Hold the right mouse button for windowing (can be reset by pressing R).\n'
                f'Press C to toggle color mode.\n'
                'Press H to hide all contours.\n'
                'Press J to jiggle around the current frame.\n'
                'Press Q to close the program.\n'
            )
        )

        dicomButton.clicked.connect(lambda _: readDICOM(self))
        gatingButton.clicked.connect(self.gate)
        segmentButton.clicked.connect(lambda _: segment(self))
        splineButton.clicked.connect(lambda _: newSpline(self))
        writeButton.clicked.connect(lambda _: writeContours(self))
        reportButton.clicked.connect(lambda _: report(self))

        self.playButton = QPushButton()
        pixmapi1 = getattr(QStyle, 'SP_MediaPlay')
        pixmapi2 = getattr(QStyle, 'SP_MediaPause')
        self.playIcon = self.style().standardIcon(pixmapi1)
        self.pauseIcon = self.style().standardIcon(pixmapi2)
        self.playButton.setIcon(self.playIcon)
        self.playButton.setMaximumWidth(30)
        self.playButton.clicked.connect(self.play)
        self.paused = True

        self.slider = Slider(Qt.Horizontal)
        self.slider.valueChanged[int].connect(self.changeValue)

        max_box_width = 130
        self.diastolicFrameBox = QCheckBox('Diastolic Frame')
        self.diastolicFrameBox.setMaximumWidth(max_box_width)
        self.diastolicFrameBox.setChecked(False)
        self.diastolicFrameBox.stateChanged[int].connect(self.toggleDiastolicFrame)
        self.systolicFrameBox = QCheckBox('Systolic Frame')
        self.systolicFrameBox.setMaximumWidth(max_box_width)
        self.systolicFrameBox.setChecked(False)
        self.systolicFrameBox.stateChanged[int].connect(self.toggleSystolicFrame)
        self.plaqueFrameBox = QCheckBox('Plaque')
        self.plaqueFrameBox.setMaximumWidth(max_box_width)
        self.plaqueFrameBox.setChecked(False)
        self.plaqueFrameBox.stateChanged[int].connect(self.togglePlaqueFrame)

        self.hideBox = QCheckBox('&Hide Contours')
        self.hideBox.setChecked(True)
        self.hideBox.stateChanged[int].connect(self.changeState)
        self.useDiastolicButton = QPushButton('Diastolic Frames')
        self.useDiastolicButton.setStyleSheet('background-color: #192f91')
        self.useDiastolicButton.setCheckable(True)
        self.useDiastolicButton.setChecked(True)
        self.useDiastolicButton.clicked.connect(self.useDiastolic)
        self.useDiastolicButton.setToolTip("Press button to switch between diastolic and systolic frames")

        self.display = Display(self, self.config)
        self.comms = Communicate()
        self.comms.updateBW[int].connect(self.display.setFrame)
        self.comms.updateBool[bool].connect(self.display.setDisplay)

        self.text = QLabel()
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setText("Frame {}".format(self.slider.value() + 1))

        vbox1.addWidget(self.display)
        vbox1hbox1.addWidget(self.playButton)
        vbox1hbox1.addWidget(self.slider)
        vbox1hbox1.addWidget(self.diastolicFrameBox)
        vbox1hbox1.addWidget(self.systolicFrameBox)
        vbox1hbox1.addWidget(self.plaqueFrameBox)
        vbox1.addLayout(vbox1hbox1)
        vbox1.addWidget(self.text)

        vbox2.addWidget(self.hideBox)
        vbox2.addWidget(self.useDiastolicButton)
        vbox2.addWidget(dicomButton)
        vbox2.addWidget(gatingButton)
        vbox2.addWidget(segmentButton)
        vbox2.addWidget(splineButton)
        vbox2.addWidget(writeButton)
        vbox2.addWidget(reportButton)
        vbox2hbox1.addWidget(self.infoTable)
        vbox2hbox2.addWidget(self.shortcutInfo)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setWindowIcon(QIcon('Media/thumbnail.png'))
        self.setWindowTitle('DeepIVUS')
        self.setCentralWidget(centralWidget)
        self.show()
        # disclaimer = QMessageBox.about(
        #     self, 'DeepIVUS', 'DeepIVUS is not FDA approved and should not be used for medical decisions.'
        # )

        # pipe = subprocess.Popen(["rm","-r","some.file"])
        # pipe.communicate() # block until process completes.
        timer = QTimer(self)
        timer.timeout.connect(self.autoSave)
        timer.start(autoSaveInterval)  # autosave interval in milliseconds

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_H:
            if self.image_displayed:
                if not self.hideBox.isChecked():
                    self.hideBox.setChecked(True)
                elif self.hideBox.isChecked():
                    self.hideBox.setChecked(False)
                self.hideBox.setChecked(self.hideBox.isChecked())
        elif key == Qt.Key_J:
            if self.image_displayed:
                currentFrame = self.slider.value()
                self.slider.setValue(currentFrame + 1)
                QApplication.processEvents()
                time.sleep(0.1)
                self.slider.setValue(currentFrame)
                QApplication.processEvents()
                time.sleep(0.1)
                self.slider.setValue(currentFrame - 1)
                QApplication.processEvents()
                time.sleep(0.1)
                self.slider.setValue(currentFrame)
                QApplication.processEvents()
        elif key == Qt.Key_W or key == Qt.Key_Up:
            self.slider.next_gated_frame()
        elif key == Qt.Key_S or key == Qt.Key_Down:
            self.slider.last_gated_frame()
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self.slider.setValue(self.slider.value() + 1)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            self.slider.setValue(self.slider.value() - 1)
        elif key == Qt.Key_E:
            if self.image_displayed:
                self.display.new_contour(self)  # start new manual Lumen contour
                self.hideBox.setChecked(False)
                self.contours_drawn = True
        if event.key() == Qt.Key.Key_R:
            # Reset window level and window width to initial values
            self.display.window_level = self.display.initial_window_level
            self.display.window_width = self.display.initial_window_width
            self.display.displayImage(update_image=True)
        elif event.key() == Qt.Key.Key_C:
            # Toggle colormap
            self.colormap_enabled = not self.colormap_enabled
            self.display.displayImage(update_image=True)

    def closeEvent(self, event):
        """Tasks to be performed before actually closing the program"""
        # if self.image and self.contours:
        #     self.save_before_close()

    def save_before_close(self):
        """Save contours, etc before closing program or reading new DICOM file"""
        self.status_bar.showMessage('Saving contours and NIfTI files...')
        writeContours(self)
        save_as_nifti(self)
        self.status_bar.showMessage('Waiting for user input')

    def play(self):
        """Plays all frames until end of pullback starting from currently selected frame"""
        start_frame = self.slider.value()

        if self.paused:
            self.paused = False
            self.playButton.setIcon(self.pauseIcon)
        else:
            self.paused = True
            self.playButton.setIcon(self.playIcon)

        for frame in range(start_frame, self.metadata['number_of_frames']):
            if not self.paused:
                self.slider.setValue(frame)
                QApplication.processEvents()
                time.sleep(0.05)
                self.text.setText("Frame {}".format(frame + 1))

        self.playButton.setIcon(self.playIcon)

    def gate(self):
        """Extract end diastolic frames and stores in new variable"""
        try:
            preprocessor = PreProcessing(self.images, self.dicom.CineRate, self.ivusPullbackRate)
            self.gated_frames_dia, self.gated_frames_sys, self.distance_frames = preprocessor()
            self.gated_frames = self.gated_frames_dia  # diastolic frames by default
        except AttributeError:  # self.images not defined because no file was read first
            warning = QErrorMessage(self)
            warning.setWindowModality(Qt.WindowModal)
            warning.showMessage("Please first read a DICOM file")
            warning.exec_()
            return

        if self.gated_frames is not None:
            self.slider.addGatedFrames(self.gated_frames)
            self.useDiastolicButton.setChecked(True)
        else:
            warning = QErrorMessage(self)
            warning.setWindowModality(Qt.WindowModal)
            warning.showMessage("Diastolic/Systolic frame extraction was unsuccessful")
            warning.exec_()

    def autoSave(self):
        """Automatically saves contours to a temporary file every autoSaveInterval seconds"""
        if self.image_displayed:
            writeContours(self)

    def changeValue(self, value):
        self.comms.updateBW.emit(value)
        self.display.run()
        self.text.setText("Frame {}".format(value + 1))
        try:
            if self.data['plaque_frames'][value] == '1':
                self.plaqueFrameBox.setChecked(True)
            else:
                self.plaqueFrameBox.setChecked(False)
        except IndexError:
            pass

        try:
            if value in self.gated_frames_dia:
                self.diastolicFrameBox.setChecked(True)
            else:
                self.diastolicFrameBox.setChecked(False)
                if value in self.gated_frames_sys:
                    self.systolicFrameBox.setChecked(True)
                else:
                    self.systolicFrameBox.setChecked(False)
        except AttributeError:
            pass

    def changeState(self, value):
        self.comms.updateBool.emit(value)
        self.display.run()

    def useDiastolic(self):
        if self.image_displayed:
            if self.useDiastolicButton.isChecked():
                self.useDiastolicButton.setText('Diastolic Frames')
                self.useDiastolicButton.setStyleSheet('background-color: #192f91')
                self.gated_frames = self.gated_frames_dia
            else:
                self.useDiastolicButton.setText('Systolic Frames')
                self.useDiastolicButton.setStyleSheet('background-color: #912519')
                self.gated_frames = self.gated_frames_sys

            self.slider.addGatedFrames(self.gated_frames)

    def toggleDiastolicFrame(self, state_true):
        if self.image_displayed:
            frame = self.slider.value()
            if state_true:
                if frame not in self.gated_frames_dia:
                    bisect.insort_left(self.gated_frames_dia, frame)
                    self.data['phases'][frame] = 'D'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.systolicFrameBox.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.gated_frames_dia.remove(frame)
                    self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            self.slider.addGatedFrames(self.gated_frames_dia)            

    def toggleSystolicFrame(self, state_true):
        if self.image_displayed:
            frame = self.slider.value()
            if state_true:
                if frame not in self.gated_frames_sys:
                    bisect.insort_left(self.gated_frames_sys, frame)
                    self.data['phases'][frame] = 'S'
                try:  # frame cannot be diastolic and systolic at the same time
                    self.diastolicFrameBox.setChecked(False)
                except ValueError:
                    pass
            else:
                try:
                    self.gated_frames_sys.remove(frame)
                    self.data['phases'][frame] = '-'
                except ValueError:
                    pass
            self.slider.addGatedFrames(self.gated_frames_sys)            

    def togglePlaqueFrame(self, state_true):
        if self.image_displayed:
            frame = self.slider.value()
            if state_true:
                self.data['plaque_frames'][frame] = '1'
            else:
                self.data['plaque_frames'][frame] = '0'

    def errorMessage(self):
        """Helper function for errors"""

        warning = QMessageBox(self)
        warning.setWindowModality(Qt.WindowModal)
        warning.setWindowTitle("Error")
        warning.setText("Segmentation must be performed first")
        warning.exec_()

    def successMessage(self, task):
        """Helper function for success messages"""

        success = QMessageBox(self)
        success.setWindowModality(Qt.WindowModal)
        success.setWindowTitle("Status")
        success.setText(task + " has been successfully completed")
        success.exec_()
