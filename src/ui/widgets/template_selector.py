"""
Template Selector Pattern for Foundation UI.

Provides polymorphic rendering for tree views and lists based on data type.
Port of CollectionTools' itemTemplateSelector pattern to Qt delegates.
"""
from typing import Dict, Type, Callable, Optional, Any
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QWidget
from PySide6.QtCore import QModelIndex, Qt
from loguru import logger


class TemplateSelector:
    """
    Maps data types to delegate/widget classes for polymorphic rendering.
    
    Usage:
        selector = TemplateSelector()
        selector.add_template(FileRecord, FileItemDelegate)
        selector.add_template(DirectoryRecord, DirectoryItemDelegate)
        selector.add_template_func(lambda item: isinstance(item, Tag), TagItemDelegate)
        
        delegate = selector.get_delegate(item)
    """
    
    def __init__(self):
        # Type -> Delegate class mappings
        self._type_templates: Dict[Type, Type[QStyledItemDelegate]] = {}
        
        # Function-based selectors (for more complex matching)
        self._func_selectors: list = []
        
        # Default delegate class
        self._default_delegate: Optional[Type[QStyledItemDelegate]] = None
    
    def add_template(
        self, 
        data_type: Type, 
        delegate_class: Type[QStyledItemDelegate]
    ) -> 'TemplateSelector':
        """
        Register a delegate for a specific data type.
        
        Args:
            data_type: The Python class to match
            delegate_class: QStyledItemDelegate subclass to use
            
        Returns:
            Self for chaining
        """
        self._type_templates[data_type] = delegate_class
        return self
    
    def add_template_func(
        self, 
        matcher: Callable[[Any], bool], 
        delegate_class: Type[QStyledItemDelegate]
    ) -> 'TemplateSelector':
        """
        Register a delegate with a custom matching function.
        
        Args:
            matcher: Function that returns True if item matches
            delegate_class: QStyledItemDelegate subclass to use
            
        Returns:
            Self for chaining
        """
        self._func_selectors.append((matcher, delegate_class))
        return self
    
    def set_default(
        self, 
        delegate_class: Type[QStyledItemDelegate]
    ) -> 'TemplateSelector':
        """Set default delegate for unmatched types."""
        self._default_delegate = delegate_class
        return self
    
    def get_delegate_class(self, item: Any) -> Optional[Type[QStyledItemDelegate]]:
        """
        Get the delegate class for an item.
        
        Matching order:
        1. Exact type match
        2. isinstance check for registered types
        3. Function-based selectors
        4. Default delegate
        
        Args:
            item: The data item to match
            
        Returns:
            Delegate class or None if no match
        """
        if item is None:
            return self._default_delegate
        
        item_type = type(item)
        
        # 1. Exact type match
        if item_type in self._type_templates:
            return self._type_templates[item_type]
        
        # 2. isinstance match (for inheritance)
        for registered_type, delegate_class in self._type_templates.items():
            if isinstance(item, registered_type):
                return delegate_class
        
        # 3. Function selectors
        for matcher, delegate_class in self._func_selectors:
            try:
                if matcher(item):
                    return delegate_class
            except Exception:
                pass
        
        # 4. Default
        return self._default_delegate
    
    def create_delegate(
        self, 
        item: Any, 
        parent: QWidget = None
    ) -> Optional[QStyledItemDelegate]:
        """
        Create a delegate instance for an item.
        
        Args:
            item: Data item to create delegate for
            parent: Parent widget for the delegate
            
        Returns:
            Delegate instance or None
        """
        delegate_class = self.get_delegate_class(item)
        if delegate_class:
            return delegate_class(parent)
        return None


class CompositeDelegate(QStyledItemDelegate):
    """
    Delegate that uses TemplateSelector to dispatch rendering.
    
    Set this as the delegate on a view, and it will automatically
    use the correct template for each item based on its type.
    """
    
    def __init__(self, selector: TemplateSelector, parent: QWidget = None):
        super().__init__(parent)
        self._selector = selector
        self._delegate_cache: Dict[Type, QStyledItemDelegate] = {}
    
    def _get_delegate_for_item(self, item: Any) -> Optional[QStyledItemDelegate]:
        """Get or create delegate for item type."""
        if item is None:
            return None
        
        item_type = type(item)
        
        # Check cache
        if item_type in self._delegate_cache:
            return self._delegate_cache[item_type]
        
        # Create new delegate
        delegate = self._selector.create_delegate(item, self.parent())
        if delegate:
            self._delegate_cache[item_type] = delegate
        
        return delegate
    
    def paint(
        self, 
        painter, 
        option: QStyleOptionViewItem, 
        index: QModelIndex
    ):
        """Delegate paint to appropriate template."""
        item = index.data(Qt.ItemDataRole.UserRole)
        delegate = self._get_delegate_for_item(item)
        
        if delegate:
            delegate.paint(painter, option, index)
        else:
            super().paint(painter, option, index)
    
    def sizeHint(
        self, 
        option: QStyleOptionViewItem, 
        index: QModelIndex
    ):
        """Delegate size hint to appropriate template."""
        item = index.data(Qt.ItemDataRole.UserRole)
        delegate = self._get_delegate_for_item(item)
        
        if delegate:
            return delegate.sizeHint(option, index)
        return super().sizeHint(option, index)
    
    def createEditor(
        self, 
        parent: QWidget, 
        option: QStyleOptionViewItem, 
        index: QModelIndex
    ):
        """Delegate editor creation to appropriate template."""
        item = index.data(Qt.ItemDataRole.UserRole)
        delegate = self._get_delegate_for_item(item)
        
        if delegate:
            return delegate.createEditor(parent, option, index)
        return super().createEditor(parent, option, index)


# Example delegates for common types
class DefaultItemDelegate(QStyledItemDelegate):
    """Default delegate with basic rendering."""
    pass


class IconTextDelegate(QStyledItemDelegate):
    """
    Delegate showing icon and text.
    
    Expects item to have 'icon' and 'text' attributes or be a tuple.
    """
    
    def paint(self, painter, option, index):
        # Get item data
        item = index.data(Qt.ItemDataRole.UserRole)
        
        if hasattr(item, 'icon') and hasattr(item, 'name'):
            # Draw icon + text
            icon = item.icon
            text = item.name
            # Custom painting logic here
            pass
        
        # Fall back to default
        super().paint(painter, option, index)
