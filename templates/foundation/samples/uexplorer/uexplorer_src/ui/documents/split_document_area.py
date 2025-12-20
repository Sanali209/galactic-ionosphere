"""
Split Document Area - Main widget for managing split file pane views.
Provides VS Code-like split functionality with tab support.
"""
from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget, QTabBar
)
from PySide6.QtCore import Qt, Signal
from loguru import logger

import sys
from pathlib import Path

# Add to path
docs_path = Path(__file__).parent
if str(docs_path) not in sys.path:
    sys.path.insert(0, str(docs_path))

from uexplorer_src.ui.documents.split_manager import SplitManager, SplitOrientation, SplitNode
from uexplorer_src.ui.documents.file_pane_document import FilePaneDocument


class DocumentTabWidget(QTabWidget):
    """Tab widget with closable tabs and drag support."""
    
    tab_close_requested = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self._on_tab_close)
    
    def _on_tab_close(self, index: int):
        """Handle tab close request."""
        widget = self.widget(index)
        if widget and hasattr(widget, 'can_close'):
            if widget.can_close():
                self.removeTab(index)
                self.tab_close_requested.emit(index)


class SplitDocumentArea(QWidget):
    """
    Main document area supporting split views.
    
    Usage:
        area = SplitDocumentArea(locator)
        area.add_pane("Left")   # First pane
        area.split_horizontal() # Split current
        area.add_pane("Right")  # Second pane
    """
    
    # Signals
    active_pane_changed = Signal(object)  # FilePaneDocument
    selection_changed = Signal(list)  # file IDs
    
    def __init__(self, locator, parent=None):
        super().__init__(parent)
        self.locator = locator
        self._split_manager = SplitManager()
        self._containers: Dict[str, DocumentTabWidget] = {}
        self._active_container_id: Optional[str] = None
        
        # Main layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Build initial layout
        self._rebuild_layout()
        
        logger.info("SplitDocumentArea initialized")
    
    def _rebuild_layout(self):
        """Rebuild widget tree from split manager."""
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._containers.clear()
        
        # Build from split tree
        root_widget = self._build_widget(self._split_manager.root)
        if root_widget:
            self._layout.addWidget(root_widget)
    
    def _build_widget(self, node: SplitNode) -> Optional[QWidget]:
        """Recursively build widgets from split tree."""
        if node.is_container:
            # Create tab container
            tab_widget = DocumentTabWidget()
            self._containers[node.id] = tab_widget
            
            # Add initial file pane
            if not node.container_widget:
                pane = FilePaneDocument(self.locator, title=f"Browser {len(self._containers)}")
                pane.selection_changed.connect(self.selection_changed.emit)
                tab_widget.addTab(pane, pane.title)
                node.container_widget = pane
            else:
                # Reuse existing widget
                tab_widget.addTab(node.container_widget, node.container_widget.title)
            
            # Set as active if first container
            if self._active_container_id is None:
                self._active_container_id = node.id
            
            return tab_widget
        else:
            # Create splitter with children
            orientation = Qt.Horizontal if node.orientation == SplitOrientation.HORIZONTAL else Qt.Vertical
            splitter = QSplitter(orientation)
            
            for child in node.children:
                child_widget = self._build_widget(child)
                if child_widget:
                    splitter.addWidget(child_widget)
            
            # Set sizes if available
            if node.sizes and len(node.sizes) == len(node.children):
                splitter.setSizes(node.sizes)
            
            return splitter
    
    def add_pane(self, title: str = None) -> FilePaneDocument:
        """Add a new file pane to the active container."""
        if not self._active_container_id:
            return None
        
        container = self._containers.get(self._active_container_id)
        if not container:
            return None
        
        title = title or f"Browser {len(self._containers) + container.count()}"
        pane = FilePaneDocument(self.locator, title=title)
        pane.selection_changed.connect(self.selection_changed.emit)
        container.addTab(pane, pane.title)
        container.setCurrentWidget(pane)
        
        logger.info(f"Added new pane: {title}")
        return pane
    
    def split_horizontal(self):
        """Split active container horizontally (side by side)."""
        self._split(SplitOrientation.HORIZONTAL)
    
    def split_vertical(self):
        """Split active container vertically (top/bottom)."""
        self._split(SplitOrientation.VERTICAL)
    
    def _split(self, orientation: SplitOrientation):
        """Split active container."""
        if not self._active_container_id:
            return
        
        new_id = self._split_manager.split_node(self._active_container_id, orientation)
        if new_id:
            self._rebuild_layout()
            self._active_container_id = new_id
            logger.info(f"Split {orientation.name}, new container: {new_id}")
    
    def close_split(self):
        """Close active split and merge with sibling."""
        if not self._active_container_id:
            return
        
        if self._split_manager.remove_split(self._active_container_id):
            self._active_container_id = None
            self._rebuild_layout()
            
            # Set first container as active
            if self._containers:
                self._active_container_id = next(iter(self._containers.keys()))
    
    def get_all_panes(self) -> list:
        """Get all file pane documents."""
        panes = []
        for container in self._containers.values():
            for i in range(container.count()):
                widget = container.widget(i)
                if isinstance(widget, FilePaneDocument):
                    panes.append(widget)
        return panes
    
    def get_active_pane(self) -> Optional[FilePaneDocument]:
        """Get currently active file pane."""
        if not self._active_container_id:
            return None
        
        container = self._containers.get(self._active_container_id)
        if container:
            widget = container.currentWidget()
            if isinstance(widget, FilePaneDocument):
                return widget
        return None
    
    def refresh_all(self):
        """Refresh all file panes."""
        for pane in self.get_all_panes():
            pane.refresh()
    
    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "split_tree": self._split_manager.to_dict(),
            "active_container_id": self._active_container_id
        }
    
    def set_state(self, state: dict):
        """Restore state from serialization."""
        if "split_tree" in state:
            self._split_manager = SplitManager.from_dict(state["split_tree"])
        self._active_container_id = state.get("active_container_id")
        self._rebuild_layout()
