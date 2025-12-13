from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QTextEdit, QPushButton
from PySide6.QtCore import Signal

class PropertiesWidget(QWidget):
    metadataChanged = Signal(str, str, str) # id, key, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_id = None
        
        self.layout = QVBoxLayout(self)
        
        self.form = QFormLayout()
        
        self.lbl_path = QLabel("-")
        self.lbl_path.setWordWrap(True)
        self.form.addRow("Path:", self.lbl_path)
        
        self.lbl_dims = QLabel("-")
        self.form.addRow("Dimensions:", self.lbl_dims)
        
        self.lbl_size = QLabel("-")
        self.form.addRow("Size:", self.lbl_size)
        
        self.txt_rating = QLineEdit()
        self.txt_rating.returnPressed.connect(self._save_rating)
        self.form.addRow("Rating:", self.txt_rating)
        
        self.txt_desc = QTextEdit()
        self.txt_desc.setMaximumHeight(100)
        self.form.addRow("Description:", self.txt_desc)
        
        self.btn_save_desc = QPushButton("Save Description")
        self.btn_save_desc.clicked.connect(self._save_desc)
        self.form.addRow(self.btn_save_desc)
        
        self.layout.addLayout(self.form)
        self.layout.addStretch()

    def set_data(self, image_id, path, dims, size, meta_json):
        self.current_image_id = image_id
        self.lbl_path.setText(path)
        self.lbl_dims.setText(dims)
        self.lbl_size.setText(size)
        
        # quick parse meta if possible, or just defaults
        # For prototype, we just clear editables unless we pull from meta
        # The bridge signal passed meta_json as string
        self.txt_rating.setText("") # TODO: extract from meta
        self.txt_desc.setText("")   # TODO: extract from meta

    def _save_rating(self):
        if self.current_image_id:
            val = self.txt_rating.text()
            self.metadataChanged.emit(self.current_image_id, "rating", val)

    def _save_desc(self):
        if self.current_image_id:
            val = self.txt_desc.toPlainText()
            self.metadataChanged.emit(self.current_image_id, "description", val)
