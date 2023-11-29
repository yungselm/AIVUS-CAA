from PyQt5.QtWidgets import QDialog, QLineEdit, QDialogButtonBox, QFormLayout


class SegmentDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.lower_limit = QLineEdit(self)
        self.lower_limit.setText('1')
        self.upper_limit = QLineEdit(self)
        self.upper_limit.setText(str(main_window.images.shape[0]))
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow('Lower limit', self.lower_limit)
        layout.addRow('Upper limit', self.upper_limit)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return int(self.lower_limit.text()) - 1, int(self.upper_limit.text())