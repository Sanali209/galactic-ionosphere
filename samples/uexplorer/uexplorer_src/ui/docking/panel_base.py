"""
Base Panel for UExplorer

Panel widgets for use with DockingService (QWidget-based, not QDockWidget).
"""
from typing import TYPE_CHECKING, Optional, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
from loguru import logger

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class PanelBase(QWidget):
    """
    Base class for UExplorer panels.
    
    Now inherits from QWidget (not QDockWidget) to work with DockingService.
    DockingService wraps these panels in CDockWidget automatically.
    """
    
    # Signals
    panel_closed = Signal()
    panel_shown = Signal()
    panel_hidden = Signal()
    
    def __init__(self, locator: "ServiceLocator", parent: Optional[QWidget] = None) -> None:
        """
        Initialize panel.
        
        Args:
            locator: ServiceLocator for accessing services
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.locator: "ServiceLocator" = locator
        self.setup_ui()
    
    def setup_ui(self):
        """Override this to setup panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def refresh(self):
        """Override this to refresh panel data."""
        pass
    
    def on_update(self, context=None):
        """
        Called when panel should refresh its content.
        Override for custom behavior.
        """
        pass
    
    def on_show(self):
        """
        Called when panel becomes visible.
        Override for custom behavior.
        """
        pass
    
    def on_hide(self):
        """
        Called when panel is hidden.
        Override for custom behavior.
        """
        pass
    
    def _on_visibility_changed(self, visible: bool):
        """Internal handler for visibility changes."""
        if visible:
            self.panel_shown.emit()
            self.on_show()
        else:
            self.panel_hidden.emit()
            self.on_hide()
    
    def get_state(self) -> dict:
        """
        Return panel-specific state for serialization.
        Override to save custom state.
        """
        return {}
    
    def set_state(self, state: dict):
        """
        Restore panel-specific state.
        Override to restore custom state.
        """
        pass
