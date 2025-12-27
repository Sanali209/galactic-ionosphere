"""
HierarchicalFilterTree - Reusable tree widget with include/exclude checkboxes.

Used by DirectoryPanel, TagPanel, AlbumPanel for unified filtering UI.

Features:
- Hierarchical tree display
- Include checkbox (green) - items matching this will be included
- Exclude checkbox (red) - items matching this will be excluded
- Tri-state propagation to children
- Keyboard shortcuts (I = include, E = exclude)

Signals:
- filter_changed(include_ids, exclude_ids): Emitted when selection changes
"""
from typing import List, Optional, Dict, Any, Set
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QHeaderView,
    QAbstractItemView, QStyledItemDelegate, QStyle
)
from PySide6.QtCore import Qt, Signal, QModelIndex, QAbstractItemModel
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QBrush, QKeyEvent
from loguru import logger


class FilterCheckDelegate(QStyledItemDelegate):
    """Custom delegate for rendering include/exclude checkboxes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def paint(self, painter, option, index):
        """Paint cell with colored checkboxes."""
        column = index.column()
        
        if column in (1, 2):  # Include or Exclude columns
            # Get check state
            value = index.data(Qt.ItemDataRole.CheckStateRole)
            is_checked = value == Qt.CheckState.Checked
            is_partial = value == Qt.CheckState.PartiallyChecked
            
            # Draw checkbox
            opt = option
            rect = option.rect
            
            # Set color based on column
            if is_checked or is_partial:
                if column == 1:  # Include - green
                    painter.fillRect(rect, QBrush(QColor(76, 175, 80, 100)))  # Light green
                else:  # Exclude - red
                    painter.fillRect(rect, QBrush(QColor(244, 67, 54, 100)))  # Light red
            
            # Let Qt draw the checkbox itself
            super().paint(painter, option, index)
        else:
            super().paint(painter, option, index)


class FilterTreeModel(QStandardItemModel):
    """Model for hierarchical filter tree."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Include", "Exclude"])
        
    def add_item(
        self,
        item_id: str,
        label: str,
        parent: Optional[QStandardItem] = None,
        icon=None
    ) -> QStandardItem:
        """Add an item to the tree."""
        # Create item for label column
        label_item = QStandardItem(label)
        label_item.setData(item_id, Qt.ItemDataRole.UserRole)
        label_item.setEditable(False)
        if icon:
            label_item.setIcon(icon)
        
        # Create checkboxes for include/exclude
        include_item = QStandardItem()
        include_item.setCheckable(True)
        include_item.setCheckState(Qt.CheckState.Unchecked)
        include_item.setEditable(False)
        include_item.setData(item_id, Qt.ItemDataRole.UserRole)
        
        exclude_item = QStandardItem()
        exclude_item.setCheckable(True)
        exclude_item.setCheckState(Qt.CheckState.Unchecked)
        exclude_item.setEditable(False)
        exclude_item.setData(item_id, Qt.ItemDataRole.UserRole)
        
        row = [label_item, include_item, exclude_item]
        
        if parent:
            parent.appendRow(row)
        else:
            self.appendRow(row)
        
        return label_item


class HierarchicalFilterTree(QWidget):
    """
    Tree widget with include/exclude checkboxes.
    
    Signals:
        filter_changed(list, list): Emitted with (include_ids, exclude_ids)
    """
    
    filter_changed = Signal(list, list)  # (include_ids, exclude_ids)
    item_clicked = Signal(str)  # item_id - for navigation
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._item_map: Dict[str, QStandardItem] = {}  # id -> label item
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tree view
        self.tree = QTreeView()
        self.tree.setHeaderHidden(False)
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Create model
        self.model = FilterTreeModel()
        self.tree.setModel(self.model)
        
        # Set delegate for custom checkbox rendering
        self.tree.setItemDelegate(FilterCheckDelegate(self.tree))
        
        # Configure header
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 60)  # Include column width
        header.resizeSection(2, 60)  # Exclude column width
        
        # Connect signals
        self.model.itemChanged.connect(self._on_item_changed)
        self.tree.clicked.connect(self._on_tree_clicked)
        
        layout.addWidget(self.tree)
    
    def clear(self):
        """Clear all items."""
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Name", "Include", "Exclude"])
        self._item_map.clear()
    
    def add_item(
        self,
        item_id: str,
        label: str,
        parent_id: Optional[str] = None,
        icon=None
    ) -> None:
        """Add an item to the tree."""
        parent = None
        if parent_id and parent_id in self._item_map:
            parent = self._item_map[parent_id]
        
        item = self.model.add_item(item_id, label, parent, icon)
        self._item_map[item_id] = item
    
    def get_include_ids(self) -> List[str]:
        """Get list of included item IDs."""
        return self._get_checked_ids(column=1)
    
    def get_exclude_ids(self) -> List[str]:
        """Get list of excluded item IDs."""
        return self._get_checked_ids(column=2)
    
    def _get_checked_ids(self, column: int) -> List[str]:
        """Get IDs of checked items in given column."""
        checked = []
        
        def traverse(parent: QStandardItem):
            for row in range(parent.rowCount() if parent else self.model.rowCount()):
                if parent:
                    label_item = parent.child(row, 0)
                    check_item = parent.child(row, column)
                else:
                    label_item = self.model.item(row, 0)
                    check_item = self.model.item(row, column)
                
                if check_item and check_item.checkState() == Qt.CheckState.Checked:
                    item_id = label_item.data(Qt.ItemDataRole.UserRole)
                    if item_id:
                        checked.append(item_id)
                
                # Traverse children
                if label_item and label_item.hasChildren():
                    traverse(label_item)
        
        traverse(None)
        return checked
    
    def _on_item_changed(self, item: QStandardItem):
        """Handle checkbox state change."""
        column = item.column()
        
        if column in (1, 2):  # Include or Exclude column
            # Emit filter changed signal
            include_ids = self.get_include_ids()
            exclude_ids = self.get_exclude_ids()
            self.filter_changed.emit(include_ids, exclude_ids)
    
    def _on_tree_clicked(self, index: QModelIndex):
        """Handle tree click - emit item_clicked for navigation."""
        if index.column() == 0:  # Label column
            item = self.model.itemFromIndex(index)
            if item:
                item_id = item.data(Qt.ItemDataRole.UserRole)
                if item_id:
                    self.item_clicked.emit(item_id)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        key = event.key()
        
        # Get selected item
        indexes = self.tree.selectedIndexes()
        if not indexes:
            super().keyPressEvent(event)
            return
        
        row = indexes[0].row()
        parent = indexes[0].parent()
        
        if key == Qt.Key.Key_I:  # Toggle Include
            if parent.isValid():
                include_item = self.model.itemFromIndex(parent).child(row, 1)
            else:
                include_item = self.model.item(row, 1)
            if include_item:
                new_state = Qt.CheckState.Unchecked if include_item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                include_item.setCheckState(new_state)
                
        elif key == Qt.Key.Key_E:  # Toggle Exclude
            if parent.isValid():
                exclude_item = self.model.itemFromIndex(parent).child(row, 2)
            else:
                exclude_item = self.model.item(row, 2)
            if exclude_item:
                new_state = Qt.CheckState.Unchecked if exclude_item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                exclude_item.setCheckState(new_state)
        else:
            super().keyPressEvent(event)
