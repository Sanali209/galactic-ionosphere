"""
MVVM ViewModel Infrastructure.

Provides a single base class for ViewModels with property change notification.
Uses BindableBase for WPF-style automatic signal emission.
"""
from typing import Any
from PySide6.QtCore import Signal

# Import from bindable - this is the single source of truth
from src.ui.mvvm.bindable import BindableProperty, BindableBase


class BaseViewModel(BindableBase):
    """
    Base class for ViewModels.
    
    Extends BindableBase for unified MVVM pattern:
    - Use BindableProperty descriptor for automatic change notification
    - propertyChanged signal inherited from BindableBase
    
    Example:
        class MyViewModel(BaseViewModel):
            nameChanged = Signal(str)
            name = BindableProperty(default="")
    """
    
    def __init__(self, locator=None):
        super().__init__()
        self.locator = locator
    
    def on_property_changed(self, property_name: str, value: Any) -> None:
        """
        Emit a property changed notification.
        
        Note: With BindableProperty, this is called automatically.
        Only use for manual property implementations.
        
        Args:
            property_name: Name of the property that changed.
            value: New value of the property.
        """
        self.propertyChanged.emit(property_name, value)

