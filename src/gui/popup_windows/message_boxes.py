from PyQt5.QtWidgets import QMessageBox


class ErrorMessage(QMessageBox):
    """Error message box"""
    def __init__(self, main_window, message='Segmentation must be performed first'):
        super().__init__(main_window)
        self.setIcon(QMessageBox.Critical)
        self.setWindowTitle('Error')
        self.setText(message)
        self.exec_()

class SuccessMessage(QMessageBox):
    """Success message box"""
    def __init__(self, main_window, task):
        super().__init__(main_window)
        self.setWindowTitle('Status')
        self.setText(task + ' has been successfully completed')
        self.exec_()
