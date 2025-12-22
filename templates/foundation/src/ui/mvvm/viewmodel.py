"""
MVVM ViewModel Infrastructure.

Provides base classes for ViewModels with property change notification.
"""
from typing import Any
from PySide6.QtCore import QObject, Signal

# Re-export from bindable for convenience
from src.ui.mvvm.bindable import BindableProperty, BindableBase


class BaseViewModel(QObject):
    """
    Base class for ViewModels.
    
    Provides infrastructure for property change notifications and
    binding to the Locator.
    
    For WPF-style binding with automatic signal emission, consider
    using `BindableBase` instead (from `src.ui.mvvm.bindable`).
    """
    
    # Generic property changed signal (property_name, value)
    propertyChanged = Signal(str, object)
    
    def __init__(self, locator=None):
        super().__init__()
        self.locator = locator
    
    def on_property_changed(self, property_name: str, value: Any) -> None:
        """
        Emit a property changed notification.
        
        Call this from property setters to notify the View of changes.
        
        Args:
            property_name: Name of the property that changed.
            value: New value of the property.
        """
        self.propertyChanged.emit(property_name, value)

