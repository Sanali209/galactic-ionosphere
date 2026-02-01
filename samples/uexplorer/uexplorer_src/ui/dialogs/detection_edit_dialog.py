"""
DetectionEditDialog - Dialog for editing detection instance metadata.
Supports hierarchical class autocomplete and group editing.
"""
from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, 
    QComboBox, QDialogButtonBox, QLabel, QDoubleSpinBox, QCompleter
)
from PySide6.QtCore import Qt, QStringListModel
from loguru import logger

class DetectionEditDialog(QDialog):
    """
    Dialog for editing detection instance metadata.
    """
    def __init__(self, locator, detection_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.locator = locator
        self.data = detection_data
        self.service = None
        
        self.setWindowTitle(f"Edit Detection: {detection_data.get('name', 'Object')}")
        self.setMinimumWidth(400)
        
        self._setup_ui()
        self._load_data()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        # Class Name
        self.class_edit = QLineEdit()
        self.class_completer = QCompleter()
        self.class_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.class_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.class_edit.setCompleter(self.class_completer)
        form.addRow("Class:", self.class_edit)
        
        # Group Name
        self.group_edit = QLineEdit()
        form.addRow("Group:", self.group_edit)
        
        # Confidence
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        form.addRow("Confidence:", self.conf_spin)
        
        layout.addLayout(form)
        
        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
        # Initialize service and fetch suggestions
        from src.ucorefs.detection.service import DetectionService
        self.service = self.locator.get_system(DetectionService)
        
        if self.service:
            import asyncio
            # We use a helper to run the async fetch in the background
            self._fetch_suggestions()

    def _fetch_suggestions(self):
        """Fetch suggestions from DB and update completer."""
        if not self.service:
            return
            
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def _fetch():
            try:
                classes = await self.service.get_class_suggestions()
                # Update UI in main thread
                from PySide6.QtCore import QMetaObject, Q_ARG
                self.class_completer.setModel(QStringListModel(classes))
            except Exception as e:
                logger.error(f"Failed to fetch class suggestions for dialog: {e}")
        
        if loop.is_running():
            asyncio.create_task(_fetch())
        else:
            loop.run_until_complete(_fetch())

    def _load_data(self):
        """Load initial data into widgets."""
        self.class_edit.setText(self.data.get('class_name', ''))
        self.group_edit.setText(self.data.get('group_name', ''))
        self.conf_spin.setValue(self.data.get('confidence', 0.0))

    def get_result(self) -> Dict[str, Any]:
        """Get updated detection data."""
        return {
            "class_name": self.class_edit.text(),
            "group_name": self.group_edit.text(),
            "confidence": self.conf_spin.value()
        }
