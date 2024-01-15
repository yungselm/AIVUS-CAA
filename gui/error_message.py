from PyQt5.QtWidgets import QMessageBox


class ErrorMessage(QMessageBox):
    """Error message box"""

    def __init__(self, main_window, message):
        super().__init__(main_window)
        self.setIcon(QMessageBox.Critical)
        self.setWindowTitle('Error')
        self.setText(message)
        self.exec_()