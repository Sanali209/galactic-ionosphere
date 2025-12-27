"""
Grid panel for displaying widgets in a scrollable grid layout.
"""
from typing import List
from PySide6.QtWidgets import (QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
                                QGridLayout, QFrame)
from PySide6.QtCore import Qt, Signal

class GridPanel(QScrollArea):
    """
    Scrollable grid layout for displaying multiple widgets.
    Useful for thumbnail galleries, icon views, etc.
    """
    item_count_changed = Signal(int)
    
    def __init__(self, columns: int = 4, parent=None):
        super().__init__(parent)
        self._columns = columns
        self._items: List[QWidget] = []
        
        # Setup scroll area
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget and layout
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(10)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.setWidget(self._container)
    
    def add_item(self, widget: QWidget):
        """Add a widget to the grid."""
        self._items.append(widget)
        self._relayout()
        self.item_count_changed.emit(len(self._items))
    
    def add_items(self, widgets: List[QWidget]):
        """Add multiple widgets to the grid."""
        self._items.extend(widgets)
        self._relayout()
        self.item_count_changed.emit(len(self._items))
    
    def clear(self):
        """Remove all items from the grid."""
        # Remove from layout
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._items.clear()
        self.item_count_changed.emit(0)
    
    def remove_item(self, widget: QWidget):
        """Remove a specific widget from the grid."""
        if widget in self._items:
            self._items.remove(widget)
            widget.deleteLater()
            self._relayout()
            self.item_count_changed.emit(len(self._items))
    
    def set_columns(self, columns: int):
        """Change the number of columns."""
        if columns < 1:
            columns = 1
        
        self._columns = columns
        self._relayout()
    
    def get_columns(self) -> int:
        """Get current column count."""
        return self._columns
    
    def get_item_count(self) -> int:
        """Get total number of items."""
        return len(self._items)
    
    def _relayout(self):
        """Reorganize items in the grid."""
        # Clear layout
        while self._grid.count():
            item = self._grid.takeAt(0)
        
        # Add items back in grid layout
        for i, widget in enumerate(self._items):
            row = i // self._columns
            col = i % self._columns
            self._grid.addWidget(widget, row, col)
