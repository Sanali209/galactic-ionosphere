"""
Base panel widget for all dockable panels.
"""
from PySide6.QtWidgets import QDockWidget, QWidget
from PySide6.QtCore import Signal
from loguru import logger

class BasePanelWidget(QDockWidget):
    """
    Base class for all panel widgets.
    Provides lifecycle hooks and consistent interface.
    """
    # Signals
    panel_shown = Signal()
    panel_hidden = Signal()
    
    def __init__(self, title: str, locator, parent=None):
        super().__init__(title, parent)
        self.locator = locator
        self.panel_name = title.lower().replace(' ', '_')
        
        # Create widget container
        self._content = QWidget()
        self.setWidget(self._content)
        
        # Track visibility
        self.visibilityChanged.connect(self._on_visibility_changed)
        
        logger.debug(f"Panel created: {self.panel_name}")
    
    def initialize_ui(self):
        """
        Override to build panel UI.
        Called once during panel creation.
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
    
    def on_update(self, context=None):
        """
        Called when panel should refresh its content.
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
