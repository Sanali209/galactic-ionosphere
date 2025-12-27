"""
Panel ViewModel - Base class for panel ViewModels.

Provides MVVM foundation for side panels that react to the active
document's context. Panels display contextual information based on
the currently active document.
"""
from typing import Any, Optional, Dict
from PySide6.QtCore import Signal

from src.ui.mvvm.bindable import BindableProperty, BindableBase


class PanelViewModel(BindableBase):
    """
    Base ViewModel for panels.
    
    Panels are supporting views (side regions) that:
    - Display context-sensitive information
    - React to active document changes
    - Can be shown/hidden/auto-hidden
    
    When the active document changes, DocumentManager calls
    `on_context_changed` with the document's content_context.
    
    Example:
        class PropertiesViewModel(PanelViewModel):
            file_name = BindableProperty(default="")
            file_size = BindableProperty(default=0)
            
            def on_context_changed(self, context):
                if hasattr(context, 'file_record'):
                    self.file_name = context.file_record.name
                    self.file_size = context.file_record.size
    """
    
    # Signals
    visibility_changed = Signal(bool)
    context_updated = Signal(object)
    
    # Bindable properties
    title = BindableProperty(default="Panel")
    is_visible = BindableProperty(default=True)
    is_loading = BindableProperty(default=False)
    
    def __init__(self, panel_id: str, locator=None):
        """
        Initialize panel ViewModel.
        
        Args:
            panel_id: Unique panel identifier
            locator: ServiceLocator for accessing services
        """
        super().__init__(locator)
        self._panel_id = panel_id
        self._context: Any = None
    
    @property
    def panel_id(self) -> str:
        """Get panel ID."""
        return self._panel_id
    
    @property
    def context(self) -> Any:
        """Get current context from active document."""
        return self._context
    
    def on_context_changed(self, context: Any) -> None:
        """
        Called when active document changes.
        
        Override to update panel content based on new context.
        
        Args:
            context: Content context from active DocumentViewModel
        """
        self._context = context
        self.context_updated.emit(context)
    
    def on_visibility_changed(self, visible: bool) -> None:
        """
        Called when panel visibility changes.
        
        Override to pause/resume operations.
        
        Args:
            visible: New visibility state
        """
        self.is_visible = visible
        self.visibility_changed.emit(visible)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get state for persistence.
        
        Returns:
            Dict with panel state
        """
        return {
            "panel_id": self._panel_id,
            "is_visible": self.is_visible,
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore panel from saved state.
        
        Args:
            state: Previously saved state dict
        """
        self.is_visible = state.get("is_visible", True)
