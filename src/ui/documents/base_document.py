"""
Base Document Widget - View component for MVVM documents.

Provides a base QWidget for document content that works with DocumentViewModel.
"""
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from loguru import logger


class BaseDocumentWidget(QWidget):
    """
    Base class for document view widgets.
    
    Works with Foundation's DocumentViewModel for MVVM pattern.
    Subclass this to create custom document editors/viewers.
    
    Example:
        class ImageViewer(BaseDocumentWidget):
            def __init__(self, viewmodel, parent=None):
                super().__init__(viewmodel, parent)
                self._setup_ui()
            
            def get_state(self) -> dict:
                return {"zoom": self._zoom_level}
            
            def set_state(self, state: dict):
                self._zoom_level = state.get("zoom", 1.0)
    """
    
    # Signals
    content_changed = Signal()
    save_requested = Signal()
    close_requested = Signal()
    state_changed = Signal()  # Emit when internal state changes
    
    def __init__(self, viewmodel=None, parent=None):
        """
        Initialize document widget.
        
        Args:
            viewmodel: DocumentViewModel instance (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        self.viewmodel = viewmodel
        self._doc_id: Optional[str] = None
        
        if viewmodel:
            self._doc_id = getattr(viewmodel, 'doc_id', None)
            logger.debug(f"BaseDocumentWidget created: {getattr(viewmodel, 'title', 'Untitled')}")
    
    @property
    def doc_id(self) -> Optional[str]:
        """Get document ID."""
        return self._doc_id or (self.viewmodel.doc_id if self.viewmodel else None)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return widget-specific state for persistence.
        
        Override in subclasses to save custom state (scroll position, 
        zoom level, selected items, etc.)
        
        Returns:
            Dict with state data
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore widget-specific state.
        
        Override in subclasses to restore custom state.
        
        Args:
            state: Previously saved state dict
        """
        pass
    
    def get_content(self) -> str:
        """Override to return current document content."""
        return ""
    
    def set_content(self, content: str) -> None:
        """Override to set document content."""
        pass
    
    def save(self) -> bool:
        """
        Save document content.
        
        Override to implement save logic. Return True on success.
        """
        if self.viewmodel:
            self.viewmodel.mark_clean()
        logger.info(f"Document saved: {self.doc_id}")
        return True
    
    def can_close(self) -> bool:
        """
        Check if document can be closed.
        
        Override to add unsaved changes prompts or validation.
        Returns True if safe to close.
        """
        if self.viewmodel:
            return self.viewmodel.can_close()
        return True
    
    def on_activated(self) -> None:
        """Called when document becomes active tab."""
        if self.viewmodel:
            self.viewmodel.on_activated()
    
    def on_deactivated(self) -> None:
        """Called when document loses focus."""
        if self.viewmodel:
            self.viewmodel.on_deactivated()
