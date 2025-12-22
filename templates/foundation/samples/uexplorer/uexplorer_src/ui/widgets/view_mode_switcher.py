"""
UExplorer - View Mode Switcher

Toolbar widget for switching between view modes.
"""
from enum import Enum
from PySide6.QtWidgets import QWidget, QHBoxLayout, QToolButton, QButtonGroup
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
from loguru import logger


class ViewMode(Enum):
    """Available view modes."""
    TREE = "tree"
    LIST = "list"
    CARD = "card"


class ViewModeSwitcher(QWidget):
    """
    Toggle buttons for switching view modes.
    
    Signals:
        mode_changed(str): Emitted when mode changes
    """
    
    mode_changed = Signal(str)  # ViewMode value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        # Tree view button
        self.btn_tree = QToolButton()
        self.btn_tree.setText("≡")
        self.btn_tree.setToolTip("Tree View")
        self.btn_tree.setCheckable(True)
        self.btn_tree.setChecked(True)
        self.btn_tree.setFixedSize(28, 28)
        self.button_group.addButton(self.btn_tree, 0)
        layout.addWidget(self.btn_tree)
        
        # List view button
        self.btn_list = QToolButton()
        self.btn_list.setText("☰")
        self.btn_list.setToolTip("List View")
        self.btn_list.setCheckable(True)
        self.btn_list.setFixedSize(28, 28)
        self.button_group.addButton(self.btn_list, 1)
        layout.addWidget(self.btn_list)
        
        # Card view button
        self.btn_card = QToolButton()
        self.btn_card.setText("▦")
        self.btn_card.setToolTip("Card Grid View")
        self.btn_card.setCheckable(True)
        self.btn_card.setFixedSize(28, 28)
        self.button_group.addButton(self.btn_card, 2)
        layout.addWidget(self.btn_card)
        
        # Apply style
        self._apply_style()
        
        # Connect
        self.button_group.idClicked.connect(self._on_button_clicked)
        
        self._current_mode = ViewMode.TREE
    
    def _apply_style(self):
        """Apply button styling."""
        style = """
            QToolButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                font-size: 14px;
                color: #ccc;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
            }
            QToolButton:checked {
                background-color: #5a7aaa;
                border-color: #7a9aca;
                color: white;
            }
        """
        self.btn_tree.setStyleSheet(style)
        self.btn_list.setStyleSheet(style)
        self.btn_card.setStyleSheet(style)
    
    def _on_button_clicked(self, button_id: int):
        """Handle button click."""
        modes = [ViewMode.TREE, ViewMode.LIST, ViewMode.CARD]
        self._current_mode = modes[button_id]
        self.mode_changed.emit(self._current_mode.value)
        logger.debug(f"View mode changed to: {self._current_mode.value}")
    
    @property
    def current_mode(self) -> ViewMode:
        """Get current view mode."""
        return self._current_mode
    
    def set_mode(self, mode: str):
        """Set view mode programmatically."""
        try:
            view_mode = ViewMode(mode)
            self._current_mode = view_mode
            
            if view_mode == ViewMode.TREE:
                self.btn_tree.setChecked(True)
            elif view_mode == ViewMode.LIST:
                self.btn_list.setChecked(True)
            elif view_mode == ViewMode.CARD:
                self.btn_card.setChecked(True)
                
        except ValueError:
            logger.warning(f"Unknown view mode: {mode}")
