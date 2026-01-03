"""
UExplorer MainViewModel

ViewModel for the main window, providing data and commands for the UI.
"""
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from src.ui.mvvm.bindable import BindableBase, BindableProperty

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class MainViewModel(BindableBase):
    """
    ViewModel for UExplorer main window.
    
    Provides:
    - Status message binding
    - Current path tracking
    - Search state
    """
    
    # Signals for specific properties
    statusMessageChanged = Signal(str)
    currentPathChanged = Signal(str)
    
    # Bindable properties
    status_message = BindableProperty(default="Ready")
    current_path = BindableProperty(default="")
    is_loading = BindableProperty(default=False)
    file_count = BindableProperty(default=0)
    
    def __init__(self, locator: "ServiceLocator") -> None:
        super().__init__(locator)
        # Reason: locator is stored by BindableBase.__init__, no need to override
    
    def set_status(self, message: str):
        """Update status bar message."""
        self.status_message = message
    
    def navigate_to(self, path: str):
        """Navigate to a new path."""
        self.current_path = path
        self.set_status(f"Navigated to: {path}")
