import os
import sys
import numpy as np
import pydicom
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt

path = '/home/yungselm/shared-drives/D:/Documents/2_Coding/Python/AAOCASeg/testreport'

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.saveAsJPG()

    def initUI(self):
        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a QVBoxLayout
        layout = QVBoxLayout(central_widget)

        # Create a QGraphicsView to display the DICOM image
        self.view = QGraphicsView(self)
        layout.addWidget(self.view)

        # Create a QGraphicsScene
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        # Load the DICOM image
        dicom_file = path
        dicom = pydicom.dcmread(dicom_file)
        self.dicom_data = dicom.pixel_array[512]

        # Normalize the DICOM data to 8-bit, for other libraries to work with format
        dicom_data_min = np.min(self.dicom_data)
        dicom_data_max = np.max(self.dicom_data)
        self.normalized_data = ((self.dicom_data - dicom_data_min) / (dicom_data_max - dicom_data_min) * 255).astype(np.uint8)

        # Convert the NumPy array to a QImage
        if len(self.normalized_data.shape) == 2:
            self.height, self.width = self.normalized_data.shape
            self.q_image = QImage(self.normalized_data.data, self.width, self.height, self.width, QImage.Format.Format_Grayscale8)

            # Create a QGraphicsPixmapItem to display the image
            self.pixmap = QPixmap.fromImage(self.q_image)
            pixmap_item = self.scene.addPixmap(self.pixmap)

            # Store initial window level and window width (full width, middle level)
            self.initial_window_level = 128 # window level is the center which determines the brightness of the image
            self.initial_window_width = 256 # window width is the range of pixel values that are displayed
            self.window_level = self.initial_window_level
            self.window_width = self.initial_window_width

            # Flag to toggle colormap
            self.colormap_enabled = False

            # Set up mouse event handling
            self.view.setMouseTracking(True)
            self.view.viewport().installEventFilter(self)

        else:
            # Handle cases where the data is not 2D (e.g., for 3D volumes)
            # You may need to implement a different strategy for visualization
            pass

        self.setGeometry(100, 100, self.width, self.height)
        self.setWindowTitle('DICOM Viewer')
        self.show()

    def eventFilter(self, obj, event):
        """Handle mouse events for adjusting window level and window width"""
        if event.type() == QtCore.QEvent.Type.MouseMove and event.buttons() == QtCore.Qt.MouseButton.RightButton:
            # value to adjust sensitivity of mouse movement
            sensitivity = 0.03 # 1 for default, below 1 for slower, above 1 for faster

            # Right-click drag for adjusting window level and window width
            dx = (event.x() - self.mouse_x) * sensitivity
            dy = (event.y() - self.mouse_y) * sensitivity

            self.window_level += dx
            self.window_width += dy

            self.updateImage()

        elif event.type() == QtCore.QEvent.Type.MouseButtonPress and event.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.mouse_x = event.x()
            self.mouse_y = event.y()

        return super().eventFilter(obj, event)

    def updateImage(self):
        """Update the displayed image with the adjusted window level and window width"""
        # Calculate lower and upper bounds for the adjusted window level and window width
        lower_bound = self.window_level - self.window_width / 2
        upper_bound = self.window_level + self.window_width / 2

        # Clip and normalize pixel values
        normalized_data = np.clip(self.normalized_data, lower_bound, upper_bound) # clip values to be within the range
        normalized_data = ((normalized_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)

        if self.colormap_enabled:
            # Apply an orange-blue colormap
            colormap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
            q_image = QImage(colormap.data, self.width, self.height, self.width * 3, QImage.Format.Format_RGB888)
        else:
            q_image = QImage(normalized_data.data, self.width, self.height, self.width, QImage.Format.Format_Grayscale8)

        self.q_image = q_image  # Update the QImage
        self.pixmap = QPixmap.fromImage(q_image)
        self.scene.clear()
        pixmap_item = self.scene.addPixmap(self.pixmap)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_R:
            # Reset window level and window width to initial values
            self.window_level = self.initial_window_level
            self.window_width = self.initial_window_width
            self.updateImage()
        elif event.key() == QtCore.Qt.Key.Key_C:
            # Toggle colormap
            self.colormap_enabled = not self.colormap_enabled
            self.updateImage()
    
    def saveAsJPG(self):
        # change numpy array to jpg
        print(os.path.join(path, 'test.jpg'))
        # remove filename from path
        new_path = os.path.dirname(path)
        cv2.imwrite(os.path.join(new_path, 'test.jpg'), self.dicom_data)
        # save in path     


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    sys.exit(app.exec_()) 