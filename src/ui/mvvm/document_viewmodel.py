"""
Document ViewModel - Base class for document ViewModels.

Provides MVVM foundation for documents in split/tabbed containers.
Each document has its own ViewModel that manages state, lifecycle,
and persistence.
"""
from typing import Any, Optional, Dict
from abc import abstractmethod
from PySide6.QtCore import Signal

from src.ui.mvvm.bindable import BindableProperty, BindableBase


class DocumentViewModel(BindableBase):
    """
    Base ViewModel for documents.
    
    Documents are the main content areas (center region) that can be:
    - Opened in tabs
    - Split horizontally/vertically
    - Tracked by DocumentManager
    
    Subclass this for specific document types:
    - BrowseViewModel (file browser)
    - ImageEditorViewModel
    - TextEditorViewModel
    
    Example:
        class MyDocumentViewModel(DocumentViewModel):
            content = BindableProperty(default="")
            
            def get_state(self) -> dict:
                return {"content": self.content}
            
            def restore_state(self, state: dict):
                self.content = state.get("content", "")
    """
    
    # Signals
    activated = Signal()
    deactivated = Signal()
    close_requested = Signal()
    dirty_changed = Signal(bool)
    
    # Bindable properties
    title = BindableProperty(default="Untitled")
    is_dirty = BindableProperty(default=False)
    
    def __init__(self, doc_id: str, locator=None):
        """
        Initialize document ViewModel.
        
        Args:
            doc_id: Unique document identifier
            locator: ServiceLocator for accessing services
        """
        super().__init__(locator)
        self._doc_id = doc_id
        self._content_context: Any = None
    
    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self._doc_id
    
    @property
    def content_context(self) -> Any:
        """
        Get content context for polymorphic panel updates.
        
        Panels receive this context to adapt their content.
        Override in subclasses to provide type-specific context.
        """
        return self._content_context
    
    @content_context.setter
    def content_context(self, value: Any):
        """Set content context and notify."""
        if self._content_context != value:
            self._content_context = value
            self.notify_property_changed("content_context", value)
    
    def on_activated(self) -> None:
        """
        Called when document becomes active.
        
        Override to perform actions when user switches to this document.
        """
        self.activated.emit()
    
    def on_deactivated(self) -> None:
        """
        Called when document loses focus.
        
        Override to pause operations or save draft state.
        """
        self.deactivated.emit()
    
    def can_close(self) -> bool:
        """
        Check if document can be closed.
        
        Returns:
            True if safe to close, False to prevent closing
        
        Override to add save prompts or validation.
        """
        return True
    
    def request_close(self) -> None:
        """Request to close this document."""
        if self.can_close():
            self.close_requested.emit()
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get serializable state for persistence.
        
        Returns:
            Dict with all state needed to restore document
        """
        return {
            "doc_id": self._doc_id,
            "title": self.title,
        }
    
    @abstractmethod
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore document from saved state.
        
        Args:
            state: Previously saved state dict
        """
        self.title = state.get("title", "Untitled")
    
    def mark_dirty(self) -> None:
        """Mark document as having unsaved changes."""
        if not self.is_dirty:
            self.is_dirty = True
            self.dirty_changed.emit(True)
    
    def mark_clean(self) -> None:
        """Mark document as saved."""
        if self.is_dirty:
            self.is_dirty = False
            self.dirty_changed.emit(False)
