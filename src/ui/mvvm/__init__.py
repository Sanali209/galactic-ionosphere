"""
MVVM Package - WPF-Style Data Binding for PySide6.

Provides:
- BindableProperty: Descriptor for auto-signaling properties.
- BindableBase: Base ViewModel with generic propertyChanged signal.
- DocumentViewModel: Base for document ViewModels.
- PanelViewModel: Base for panel ViewModels.
- bind(): Declarative property binding between ViewModel and View.
- DataContextMixin: ViewModel inheritance down the widget tree.
"""
from src.ui.mvvm.viewmodel import BaseViewModel, BindableProperty, BindableBase
from src.ui.mvvm.binding import bind, bind_command, BindingMode
from src.ui.mvvm.data_context import DataContextMixin, BindableWidget
from src.ui.mvvm.provider import ViewModelProvider
from src.ui.mvvm.document_viewmodel import DocumentViewModel
from src.ui.mvvm.panel_viewmodel import PanelViewModel

__all__ = [
    # ViewModels
    "BaseViewModel",
    "BindableBase",
    "BindableProperty",
    "DocumentViewModel",
    "PanelViewModel",
    
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

