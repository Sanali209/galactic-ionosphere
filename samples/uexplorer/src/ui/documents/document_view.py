"""
Document management base classes.
Provides foundation for document views with MVVM support.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from loguru import logger

class DocumentViewModel:
    """
    Base ViewModel for document state management.
    Tracks file path, modified state, content.
    """
    def __init__(self, locator, file_path: Optional[str] = None):
        self.locator = locator
        self._file_path = file_path
        self._is_modified = False
        self._title = file_path.split('/')[-1] if file_path else "Untitled"
    
    @property
    def file_path(self) -> Optional[str]:
        return self._file_path
    
    @property
    def is_modified(self) -> bool:
        return self._is_modified
    
    @property
    def title(self) -> str:
        return f"{self._title}{'*' if self._is_modified else ''}"
    
    def mark_modified(self):
        self._is_modified = True
    
    def mark_saved(self):
        self._is_modified = False

class DocumentView(QWidget):
    """
    Base class for all document types (text, image, etc.).
    Subclass this to create custom document editors/viewers.
    """
    # Signals
    content_changed = Signal()
    save_requested = Signal()
    close_requested = Signal()
    
    def __init__(self, viewmodel: DocumentViewModel, parent=None):
        super().__init__(parent)
        self.viewmodel = viewmodel
        logger.debug(f"DocumentView created: {viewmodel.title}")
    
    def get_content(self) -> str:
        """Override to return current document content."""
        return ""
    
    def set_content(self, content: str):
        """Override to set document content."""
        pass
    
    def save(self):
        """Override to implement save logic."""
        self.viewmodel.mark_saved()
        logger.info(f"Document saved: {self.viewmodel.title}")
    
    def can_close(self) -> bool:
        """Override to check if doc can close (e.g., prompt for unsaved)."""
        return True
