from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt


class SmallDisplay(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

        self.setWindowTitle("Small Display")
        self.setGeometry(100, 100, 400, 400)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowStaysOnTopHint
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
