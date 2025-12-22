"""
UExplorer - Search Panel

Search input with quick filter buttons.
Integrates with FilterManager for centralized filtering.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QLabel, QButtonGroup, QToolButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from loguru import logger


class SearchPanel(QWidget):
    """
    Search panel with text input and quick filters.
    
    Features:
    - Search input with clear button
    - Quick filter buttons (All, Images, Video, Audio, Docs)
    - Connected to FilterManager
    """
    
    search_changed = Signal(str)  # Search text
    filter_changed = Signal(str)  # Filter type
    
    def __init__(self, filter_manager=None, parent=None):
        super().__init__(parent)
        self._filter_manager = filter_manager
        self._current_filter = "all"
        
        self.setup_ui()
        self._connect_manager()
        
        logger.info("SearchPanel initialized")
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Search icon/label
        search_label = QLabel("🔍")
        layout.addWidget(search_label)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search files...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(200)
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._on_search_submit)
        layout.addWidget(self.search_input, 1)
        
        layout.addSpacing(16)
        
        # Quick filter buttons
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        filter_buttons = [
            ("All", "all", True),
            ("Images", "image", False),
            ("Video", "video", False),
            ("Audio", "audio", False),
            ("Docs", "document", False),
        ]
        
        for label, filter_type, checked in filter_buttons:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.setProperty("filter_type", filter_type)
            btn.setMinimumWidth(60)
            self.button_group.addButton(btn)
            layout.addWidget(btn)
        
        self.button_group.buttonClicked.connect(self._on_filter_button_clicked)
        
        # Apply style
        self._apply_style()
    
    def _apply_style(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #5a8aca;
            }
            QToolButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
            }
            QToolButton:checked {
                background-color: #5a7aaa;
                color: white;
                border-color: #7a9aca;
            }
            QLabel {
                color: #cccccc;
                font-size: 16px;
            }
        """)
    
    def _connect_manager(self):
        """Connect to FilterManager if available."""
        if self._filter_manager:
            self._filter_manager.search_changed.connect(self._on_external_search)
    
    def set_filter_manager(self, manager):
        """Set FilterManager reference."""
        self._filter_manager = manager
        self._connect_manager()
    
    def _on_search_changed(self, text: str):
        """Handle search text change."""
        self.search_changed.emit(text)
        
        if self._filter_manager:
            self._filter_manager.set_search_text(text)
    
    def _on_search_submit(self):
        """Handle Enter key in search."""
        text = self.search_input.text()
        self.search_changed.emit(text)
    
    def _on_filter_button_clicked(self, button):
        """Handle quick filter button click."""
        filter_type = button.property("filter_type")
        self._current_filter = filter_type
        self.filter_changed.emit(filter_type)
        
        if self._filter_manager:
            if filter_type == "all":
                self._filter_manager.clear_file_types()
            else:
                self._filter_manager.set_file_types([filter_type])
        
        logger.debug(f"Quick filter: {filter_type}")
    
    def _on_external_search(self, text: str):
        """Handle search changes from FilterManager."""
        if self.search_input.text() != text:
            self.search_input.setText(text)
    
    def clear(self):
        """Clear search and reset filters."""
        self.search_input.clear()
        # Select "All" button
        for btn in self.button_group.buttons():
            if btn.property("filter_type") == "all":
                btn.setChecked(True)
                break
        self._current_filter = "all"
