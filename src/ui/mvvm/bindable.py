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
        coerce: Optional[Callable[[Any], T]] = None
    ):
        self.default = default
        self._signal_name = signal_name
        self.coerce = coerce
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
