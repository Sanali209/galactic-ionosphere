"""
WPF-Style Bindable Property Descriptor.

Provides automatic signal emission on property change, reducing MVVM boilerplate.

Usage:
    class MyViewModel(BindableBase):
        username = BindableProperty(default="")
        age = BindableProperty(default=0)
    
    # Changing the property auto-emits usernameChanged signal
    vm.username = "Alice"
"""
from typing import Any, Optional, Callable, TypeVar, Generic
from PySide6.QtCore import QObject, Signal

T = TypeVar('T')


class BindableProperty(Generic[T]):
    """
    Descriptor that emits a signal when the property value changes.
    
    Inspired by WPF's DependencyProperty / INotifyPropertyChanged pattern.
    
    Args:
        default: Default value for the property.
        signal_name: Optional custom signal name. Defaults to "{property_name}Changed".
        coerce: Optional callable to coerce/validate the value before setting.
    
    Example:
        class UserViewModel(BindableBase):
            name = BindableProperty(default="")
            age = BindableProperty(default=0, coerce=lambda x: max(0, int(x)))
    """
    
    def __init__(
        self, 
        default: T = None, 
        signal_name: Optional[str] = None,
        coerce: Optional[Callable[[Any], T]] = None,
        sync_channel: Optional[str] = None,
        mapper: Optional[Callable[[Any], Any]] = None
    ):
        self.default = default
        self._signal_name = signal_name
        self.coerce = coerce
        self.sync_channel = sync_channel
        self.mapper = mapper
        self._attr_name: str = ""
        self._public_name: str = ""
    
    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self._public_name = name
        self._attr_name = f"_bindable_{name}"
        
        # Generate signal name if not provided
        if not self._signal_name:
            self._signal_name = f"{name}Changed"
        
        # Note: We cannot dynamically add Signals to a QObject class after
        # class creation in PySide6. Signals must be class-level attributes.
        # Users must define the signal themselves or use BindableBase which
        # provides a generic propertyChanged signal.
    
    def __get__(self, obj: Optional[QObject], objtype: type = None) -> T:
        """Get the property value."""
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self._attr_name, self.default)
    
    def __set__(self, obj: QObject, value: Any) -> None:
        """Set the property value and emit change signal if different."""
        # Check if we are in an internal update (ContextSyncManager update)
        # to prevent infinite recursion/echoes.
        is_internal = getattr(obj, "_mvvm_internal_update", False)
        
        # Coerce value if coercion function provided
        if self.coerce is not None:
            value = self.coerce(value)
        
        old_value = getattr(obj, self._attr_name, self.default)
        
        if old_value != value:
            setattr(obj, self._attr_name, value)
            
            # Try to emit specific signal first
            specific_signal = getattr(obj, self._signal_name, None)
            if specific_signal is not None and callable(getattr(specific_signal, 'emit', None)):
                specific_signal.emit(value)
            
            # Also emit generic propertyChanged if available (BindableBase)
            generic_signal = getattr(obj, 'propertyChanged', None)
            if generic_signal is not None and callable(getattr(generic_signal, 'emit', None)):
                generic_signal.emit(self._public_name, value)
            
            # --- Global Synchronization ---
            # If this property is linked to a sync channel and this wasn't an 
            # internal update, publish the change to the ContextSyncManager.
            if self.sync_channel and not is_internal:
                locator = getattr(obj, 'locator', None)
                if locator:
                    try:
                        # Lazy import to avoid circular dependency
                        from src.ui.mvvm.sync_manager import ContextSyncManager
                        sync_mgr = locator.get_system(ContextSyncManager)
                        if sync_mgr:
                            # Apply value mapper if provided (e.g. ObjectId -> str)
                            sync_value = self.mapper(value) if self.mapper else value
                            sync_mgr.publish(self.sync_channel, sync_value, source_vm=obj)
                    except (ImportError, AttributeError, KeyError):
                        # Gracefully handle if SyncManager is not yet available/registered
                        pass


class BindableBase(QObject):
    """
    Base class for ViewModels with WPF-style property change notification.
    
    Provides:
    - A generic `propertyChanged` signal for any property changes.
    - Works with `BindableProperty` descriptors for automatic notification.
    
    Example:
        class MainViewModel(BindableBase):
            # Define signals for specific properties you want to bind
            usernameChanged = Signal(str)
            
            # Use BindableProperty for automatic change notification
            username = BindableProperty(default="")
            
            def __init__(self, locator):
                super().__init__(locator)
    """
    
    # Generic signal emitted for any property change: (property_name, new_value)
    propertyChanged = Signal(str, object)
    
    def __init__(self, locator=None):
        super().__init__()
        self.locator = locator
    
    def notify_property_changed(self, property_name: str, value: Any) -> None:
        """
        Manually emit a property changed notification.
        
        Use this for properties not using BindableProperty descriptor.
        """
        self.propertyChanged.emit(property_name, value)


class BindableList(BindableBase):
    """
    A reactive list-like object that notifies when items change.
    
    Wraps a Python list and emits collectionChanged signals on mutation.
    Uses composition to avoid QObject layout conflicts with built-in list.
    """
    collectionChanged = Signal(object)
    
    def __init__(self, items=None, locator=None):
        super().__init__(locator)
        self._data = items if isinstance(items, list) else list(items or [])

    def _notify(self):
        self.collectionChanged.emit(self)

    # --- List Interface ---
    
    def __len__(self): return len(self._data)
    def __getitem__(self, i): return self._data[i]
    def __iter__(self): return iter(self._data)
    def __repr__(self): return repr(self._data)
    def __eq__(self, other): return self._data == other

    def append(self, item):
        self._data.append(item)
        self._notify()

    def extend(self, iterable):
        self._data.extend(iterable)
        self._notify()

    def insert(self, index, item):
        self._data.insert(index, item)
        self._notify()

    def remove(self, item):
        self._data.remove(item)
        self._notify()

    def pop(self, index=-1):
        item = self._data.pop(index)
        self._notify()
        return item

    def clear(self):
        self._data.clear()
        self._notify()

    def __setitem__(self, key, value):
        self._data[key] = value
        self._notify()

    def __delitem__(self, key):
        del self._data[key]
        self._notify()


class BindableDict(BindableBase):
    """
    A reactive dictionary-like object that notifies when items change.
    
    Wraps a Python dict and emits collectionChanged signals on mutation.
    Uses composition to avoid QObject layout conflicts with built-in dict.
    """
    collectionChanged = Signal(object)
    
    def __init__(self, data=None, locator=None):
        super().__init__(locator)
        self._data = data if isinstance(data, dict) else dict(data or {})

    def _notify(self):
        self.collectionChanged.emit(self)

    # --- Dict Interface ---
    
    def __getitem__(self, key): return self._data[key]
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def __repr__(self): return repr(self._data)
    def __eq__(self, other): return self._data == other
    def keys(self): return self._data.keys()
    def values(self): return self._data.values()
    def items(self): return self._data.items()
    def get(self, key, default=None): return self._data.get(key, default)

    def __setitem__(self, key, value):
        self._data[key] = value
        self._notify()

    def __delitem__(self, key):
        del self._data[key]
        self._notify()

    def update(self, *args, **kwargs):
        self._data.update(*args, **kwargs)
        self._notify()

    def pop(self, key, default=None):
        item = self._data.pop(key, default)
        self._notify()
        return item

    def clear(self):
        self._data.clear()
        self._notify()
