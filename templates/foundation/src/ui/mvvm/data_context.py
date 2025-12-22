"""
WPF-Style DataContext Propagation.

Allows ViewModels to be set on a parent widget and automatically inherited
by child widgets, similar to WPF's DataContext.

Usage:
    class MyWindow(DataContextMixin, QMainWindow):
        def __init__(self, vm):
            super().__init__()
            self.set_data_context(vm)
    
    # In a child widget:
    vm = self.get_data_context()
"""
from typing import Optional, TypeVar, TYPE_CHECKING
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QObject

if TYPE_CHECKING:
    from src.ui.mvvm.bindable import BindableBase

T = TypeVar('T', bound='BindableBase')


class DataContextMixin:
    """
    Mixin that provides WPF-style DataContext inheritance.
    
    When a ViewModel is set on a parent widget, child widgets can
    access it without explicit passing.
    
    This pattern is useful for complex UIs where deeply nested widgets
    need access to the ViewModel without threading it through constructors.
    """
    
    _data_context: Optional['BindableBase'] = None
    
    def set_data_context(self, vm: 'BindableBase', propagate: bool = True) -> None:
        """
        Set the DataContext (ViewModel) for this widget.
        
        Args:
            vm: The ViewModel to set.
            propagate: If True, recursively set on children that don't 
                       have their own DataContext.
        """
        self._data_context = vm
        
        if propagate and isinstance(self, QWidget):
            for child in self.findChildren(QWidget):
                # Only set if child is DataContextMixin and doesn't have its own
                if isinstance(child, DataContextMixin):
                    if child._data_context is None:
                        child._data_context = vm
    
    def get_data_context(self) -> Optional['BindableBase']:
        """
        Get the DataContext (ViewModel) for this widget.
        
        Walks up the widget tree to find the nearest DataContext.
        
        Returns:
            The ViewModel, or None if not found.
        """
        # Check self first
        if self._data_context is not None:
            return self._data_context
        
        # Walk up the tree
        if isinstance(self, QWidget):
            parent = self.parentWidget()
            while parent is not None:
                if isinstance(parent, DataContextMixin) and parent._data_context is not None:
                    return parent._data_context
                parent = parent.parentWidget()
        
        return None
    
    def get_typed_data_context(self, vm_type: type[T]) -> Optional[T]:
        """
        Get the DataContext cast to a specific ViewModel type.
        
        Args:
            vm_type: The expected ViewModel class.
        
        Returns:
            The ViewModel if it matches the type, else None.
        """
        ctx = self.get_data_context()
        if isinstance(ctx, vm_type):
            return ctx
        return None


class BindableWidget(DataContextMixin, QWidget):
    """
    A QWidget with built-in DataContext support.
    
    Use this as a base class for widgets that need automatic
    ViewModel inheritance.
    
    Example:
        class MyPanel(BindableWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                # ViewModel is automatically available
                vm = self.get_data_context()
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_context = None
