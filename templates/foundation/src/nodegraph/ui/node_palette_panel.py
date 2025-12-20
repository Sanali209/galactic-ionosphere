# -*- coding: utf-8 -*-
"""
NodePalettePanel - Displays available nodes for drag-and-drop.

Shows nodes organized by category, searchable.
"""
from typing import Optional, Dict, List, TYPE_CHECKING
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QTreeWidget, 
    QTreeWidgetItem, QLabel, QScrollArea, QFrame, QApplication
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QFont, QColor, QDrag, QPixmap, QPainter
from loguru import logger

if TYPE_CHECKING:
    from ..core.registry import NodeRegistry
    from ..core.base_node import BaseNode


class DraggableTreeWidget(QTreeWidget):
    """Tree widget that supports dragging nodes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self._drag_start_pos = None
    
    def mousePressEvent(self, event):
        """Track drag start position."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Start drag if moved far enough."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        
        if self._drag_start_pos is None:
            return
        
        # Check if moved far enough to start drag
        if ((event.pos() - self._drag_start_pos).manhattanLength() 
                < QApplication.startDragDistance()):
            return
        
        # Get the item being dragged
        item = self.itemAt(self._drag_start_pos)
        if item is None:
            return
        
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if not node_type:
            return  # Category item, not a node
        
        # Create drag
        drag = QDrag(self)
        
        # Set MIME data
        mime_data = QMimeData()
        mime_data.setText(node_type)
        mime_data.setData("application/x-nodetype", node_type.encode())
        drag.setMimeData(mime_data)
        
        # Create drag pixmap
        pixmap = QPixmap(120, 30)
        pixmap.fill(QColor(60, 60, 60))
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, item.text(0))
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(60, 15))
        
        # Execute drag
        drag.exec(Qt.DropAction.CopyAction)


class NodePalettePanel(QWidget):
    """
    Panel showing available nodes organized by category.
    
    Features:
    - Category tree structure
    - Search/filter
    - Double-click or drag to add node
    
    Signals:
        node_requested: Emitted when node should be added (node_type)
    """
    
    node_requested = Signal(str)  # node_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry: Optional['NodeRegistry'] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Node Palette")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Help text
        help_label = QLabel("Double-click or drag to add")
        help_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(help_label)
        
        # Search box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter_nodes)
        layout.addWidget(self._search)
        
        # Node tree (use our draggable version)
        self._tree = DraggableTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(15)
        self._tree.setAnimated(True)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)
    
    def set_registry(self, registry: 'NodeRegistry'):
        """
        Set the node registry to display.
        
        Args:
            registry: NodeRegistry with available nodes
        """
        self._registry = registry
        self._rebuild_tree()
    
    def _rebuild_tree(self):
        """Rebuild the node tree from registry."""
        self._tree.clear()
        
        if not self._registry:
            return
        
        categories = self._registry.get_categories()
        
        for category, nodes in sorted(categories.items()):
            # Create category item
            cat_item = QTreeWidgetItem([category])
            cat_item.setExpanded(True)
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            
            # Add nodes
            for node_cls in sorted(nodes, key=lambda n: n.metadata.display_name or n.node_type):
                name = node_cls.metadata.display_name or node_cls.node_type
                node_item = QTreeWidgetItem([name])
                node_item.setData(0, Qt.ItemDataRole.UserRole, node_cls.node_type)
                
                # Color indicator
                color = QColor(node_cls.metadata.color)
                node_item.setForeground(0, color)
                
                # Tooltip
                if node_cls.metadata.description:
                    node_item.setToolTip(0, node_cls.metadata.description)
                
                cat_item.addChild(node_item)
            
            self._tree.addTopLevelItem(cat_item)
    
    def _filter_nodes(self, query: str):
        """
        Filter visible nodes by search query.
        
        Args:
            query: Search text
        """
        query_lower = query.lower()
        
        for cat_idx in range(self._tree.topLevelItemCount()):
            cat_item = self._tree.topLevelItem(cat_idx)
            cat_visible = False
            
            for node_idx in range(cat_item.childCount()):
                node_item = cat_item.child(node_idx)
                name = node_item.text(0).lower()
                node_type = node_item.data(0, Qt.ItemDataRole.UserRole).lower()
                
                visible = not query or query_lower in name or query_lower in node_type
                node_item.setHidden(not visible)
                
                if visible:
                    cat_visible = True
            
            cat_item.setHidden(not cat_visible)
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on node item."""
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type:
            self.node_requested.emit(node_type)
