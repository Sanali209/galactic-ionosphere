"""
File Pane Document - Wraps FilePaneWidget as a document for split views.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
from loguru import logger

import sys
from pathlib import Path
widgets_path = Path(__file__).parent.parent / "widgets"
if str(widgets_path) not in sys.path:
    sys.path.insert(0, str(widgets_path))

from uexplorer_src.ui.widgets.file_pane import FilePaneWidget


class FilePaneDocument(QWidget):
    """
    Wraps FilePaneWidget as a document for use in split containers.
    Each file pane becomes a "document" that can be in a tabbed container.
    """
    # Signals
    content_changed = Signal()
    selection_changed = Signal(list)  # list of file IDs
    path_changed = Signal(str)  # current path
    
    def __init__(self, locator, title: str = "Browser", parent=None):
        super().__init__(parent)
        self.locator = locator
        self._title = title
        
        # Generate unique ID for drag & drop
        import uuid
        self.id = str(uuid.uuid4())
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create file pane widget (doesn't accept parent in constructor)
        self.file_pane = FilePaneWidget(locator)
        self.file_pane.setParent(self)
        layout.addWidget(self.file_pane)
        
        # Forward signals - connect file pane's signal to our handler
        self.file_pane.selection_changed.connect(self._on_selection_changed)
        
        logger.debug(f"FilePaneDocument created: {title} (ID: {self.id})")
    
    @property
    def title(self) -> str:
        """Return document title based on current path."""
        if hasattr(self.file_pane, 'current_path') and self.file_pane.current_path:
            path = Path(self.file_pane.current_path)
            return path.name or str(path)
        return self._title
    
    def _on_selection_changed(self, record_ids):
        """Handle selection change in file pane."""
        self.selection_changed.emit(record_ids)
        self.content_changed.emit()
    
    def refresh(self):
        """Refresh file pane."""
        if hasattr(self.file_pane, 'refresh_roots'):
            import asyncio
            asyncio.ensure_future(self.file_pane.refresh_roots())
    
    def can_close(self) -> bool:
        """File pane can always be closed."""
        return True
    
    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "title": self._title,
            "current_path": getattr(self.file_pane, 'current_path', None)
        }
    
    def set_state(self, state: dict):
        """Restore state from serialization."""
        self._title = state.get("title", "Browser")
        # Path restoration would go here
