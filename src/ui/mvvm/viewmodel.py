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
    
    def initialize_reactivity(self) -> None:
        """
        Initialize MVVM features for this ViewModel.
        
        Currently responsible for:
        1. Scanning for BindableProperties with 'sync_channel'
        2. Registering them with the global ContextSyncManager
        """
        if not self.locator:
            return
            
        try:
            # Lazy import to avoid circular dependencies
            from src.ui.mvvm.sync_manager import ContextSyncManager
            sync_mgr = self.locator.get_system(ContextSyncManager)
            
            if not sync_mgr:
                return
                
            # Scan class attributes for BindableProperty descriptors that joined a channel
            for attr_name in dir(self.__class__):
                # We need to access the descriptor on the class, not the value on the instance
                attr = getattr(self.__class__, attr_name)
                
                if isinstance(attr, BindableProperty) and attr.sync_channel:
                    sync_mgr.register(attr.sync_channel, self, attr_name)
                    
        except (ImportError, AttributeError):
            # Gracefully handle if SyncManager is not registered or unavailable
            pass

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

