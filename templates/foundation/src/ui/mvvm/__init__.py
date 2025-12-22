"""
MVVM Package - WPF-Style Data Binding for PySide6.

Provides:
- BindableProperty: Descriptor for auto-signaling properties.
- BindableBase: Base ViewModel with generic propertyChanged signal.
- bind(): Declarative property binding between ViewModel and View.
- DataContextMixin: ViewModel inheritance down the widget tree.
"""
from src.ui.mvvm.viewmodel import BaseViewModel, BindableProperty, BindableBase
from src.ui.mvvm.binding import bind, bind_command, BindingMode
from src.ui.mvvm.data_context import DataContextMixin, BindableWidget
from src.ui.mvvm.provider import ViewModelProvider

__all__ = [
    # ViewModels
    "BaseViewModel",
    "BindableBase",
    "BindableProperty",
    
    # Binding
    "bind",
    "bind_command",
    "BindingMode",
    
    # DataContext
    "DataContextMixin",
    "BindableWidget",
    
    # Provider
    "ViewModelProvider",
]
