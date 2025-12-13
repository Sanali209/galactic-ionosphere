from PySide6.QtWidgets import QWidget, QVBoxLayout, QListView, QAbstractItemView, QLabel
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction

class GalleryWidget(QWidget):
    selectionChanged = Signal(str) # image_id

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self.view = QListView()
        self.view.setModel(self._model)
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSpacing(10)
        self.view.setUniformItemSizes(True)
        self.view.setGridSize(QSize(220, 220))
        self.view.setIconSize(QSize(200, 200))
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Connect selection
        self.view.selectionModel().currentChanged.connect(self._on_selection_changed)
        
        self.layout.addWidget(self.view)
        
        # Status / count
        self.lbl_status = QLabel("Ready")
        self.layout.addWidget(self.lbl_status)
        
        # Connect model signals
        if hasattr(self._model, "countChanged"):
            self._model.countChanged.connect(self._update_status)

    def _on_selection_changed(self, current, previous):
        if not current.isValid():
            return
        
        # Get ID from model
        # Assuming model supports IdRole or we get it from data
        # The existing GalleryGridModel has IdRole = Qt.UserRole + 1
        idx = current
        image_id = self._model.data(idx, self._model.IdRole)
        if image_id:
            self.selectionChanged.emit(image_id)

    def _update_status(self):
        count = self._model.rowCount()
        self.lbl_status.setText(f"Images: {count}")
