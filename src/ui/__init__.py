"""
Foundation UI Framework.

Provides user interface components and patterns:
- MainWindow: Base application window with menus, docking, and MVVM support
- MVVM: Data binding and ViewModel infrastructure
- Docking: Panel and document management via QtAds
- Menus: Action registry and menu builder

Usage:
    from src.ui import MainWindow
    from src.ui.mvvm import bind, BindingMode, BaseViewModel
    from src.ui.docking import DockingService, BasePanelWidget
"""
from .main_window import MainWindow
from .docking import DockingService, BasePanelWidget
from .menus.action_registry import ActionRegistry
from .menus.menu_builder import MenuBuilder
from .mvvm import (
    bind,
    bind_command,
    BindingMode,
    BaseViewModel,
    BindableBase,
    BindableProperty,
    ViewModelProvider,
    DataContextMixin,
    BindableWidget,
)

__all__ = [
    # Main window
    "MainWindow",
    
    # Docking
    "DockingService",
    "BasePanelWidget",
    
    # Menus
    "ActionRegistry",
    "MenuBuilder",
    
    # MVVM
    "bind",
    "bind_command",
    "BindingMode",
    "BaseViewModel",
    "BindableBase",
    "BindableProperty",
    "ViewModelProvider",
    "DataContextMixin",
    "BindableWidget",
]
