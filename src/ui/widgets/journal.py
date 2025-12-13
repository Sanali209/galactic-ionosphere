from PySide6.QtWidgets import QWidget, QVBoxLayout, QListView, QLabel
from PySide6.QtCore import Qt

class JournalWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self.header = QLabel("System Log")
        self.layout.addWidget(self.header)
        
        self.view = QListView()
        self.view.setModel(self._model)
        # Use a delegate or just simple display?
        # JournalViewModel has multiple roles. DisplayRole is None in existing code?
        # Let's check JournalViewModel.data (Step 25).
        # data returns None for default roles!
        # It handles LevelRole, CategoryRole... 
        # I need to fix JournalViewModel to support DisplayRole OR use a Delegate.
        # Ideally, fix ViewModel to return a formatted string for DisplayRole.
        
        self.layout.addWidget(self.view)

    # Note: I need to update JournalViewModel to support DisplayRole
