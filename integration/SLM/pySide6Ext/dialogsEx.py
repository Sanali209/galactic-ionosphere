from PySide6.QtWidgets import QDialog, QTextEdit, QDialogButtonBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit


class UniDialog(QDialog):
    """for simple dialogs with ok and cancel buttons
    for attach to main layout use self.layout - they are QVBoxLayout"""
    def __init__(self, window_title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.ok_button = QDialogButtonBox(QDialogButtonBox.Ok)
        self.ok_button.accepted.connect(self.accept)
        self.cancel_button = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.cancel_button.rejected.connect(self.reject)
        self.ok_button.accepted.connect(self.accept)
        self.cancel_button.rejected.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        self.layout = QVBoxLayout()
        self.dialog_result = False
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def accept(self):
        self.dialog_result = True
        super().accept()
        self.close()

    def reject(self):
        self.dialog_result = False
        super().reject()
        self.close()



class StringEditor(QDialog):
    def __init__(self, window_title, initial_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.text_edit = QLineEdit(initial_text)
        self.ok_button = QDialogButtonBox(QDialogButtonBox.Ok)
        self.cancel_button = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.ok_button.accepted.connect(self.accept)
        self.cancel_button.rejected.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Edit"))
        layout.addWidget(self.text_edit)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_edited_string(self):
        return self.text_edit.text()

def show_string_editor(window_title, initial_text, parent=None):
    dialog = StringEditor(window_title, initial_text, parent)
    result = dialog.exec()
    return dialog.get_edited_string(), result == QDialog.Accepted
