from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class SmallDisplay(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setParent(main_window)
        self.image_size = main_window.config.display.image_size
        self.window_size = int(self.image_size / 1.5)
        self.resize(self.window_size, self.window_size)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
        )

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def set_frame(self, frame):
        if frame is None:
            self.pixmap.setPixmap(QPixmap())
            self.setWindowTitle("No Frame to Display")
            return

        self.pixmap.setPixmap(
            QPixmap.fromImage(
                QImage(
                    self.main_window.images[frame],
                    self.main_window.images[frame].shape[1],
                    self.main_window.images[frame].shape[0],
                    self.main_window.images[frame].shape[1],
                    QImage.Format_Grayscale8,
                ).scaled(self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            )
        )
        
        current_phase = 'Diastolic' if self.main_window.use_diastolic_button.isChecked() else 'Systolic'
        self.setWindowTitle(f"Next {current_phase} Frame {frame + 1}")
