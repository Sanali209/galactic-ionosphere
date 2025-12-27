"""
TemplateSelector - Routes items to appropriate card templates.

Maps item types to template classes for rendering.
"""
from typing import Callable, Dict, Type, TYPE_CHECKING
from loguru import logger

from src.ui.cardview.card_item_widget import CardItemWidget

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class TemplateSelector:
    """
    Selects card template based on item type.
    
    Maps item_type values to template classes, allowing
    different card layouts for different content types.
    
    Features:
    - Type → template class mapping
    - Default fallback template
    - Custom selector function support
    
    Example:
        selector = TemplateSelector()
        selector.register("image", ImageCardTemplate)
        selector.register("document", DocumentCardTemplate)
        selector.set_default(DefaultCardTemplate)
        
        template_class = selector.select(item)
        widget = template_class()
    """
    
    def __init__(self):
        """Initialize selector."""
        self._templates: Dict[str, Type[CardItemWidget]] = {}
        self._default_template: Type[CardItemWidget] = CardItemWidget
        self._custom_selector: Callable[['CardItem'], Type[CardItemWidget]] | None = None
    
    def register(self, item_type: str, template_class: Type[CardItemWidget]):
        """
        Register template for item type.
        
        Args:
            item_type: Value of CardItem.item_type to match
            template_class: Template class to instantiate
        """
        self._templates[item_type] = template_class
        logger.debug(f"Registered template for type '{item_type}'")
    
    def unregister(self, item_type: str):
        """Remove template for item type."""
        self._templates.pop(item_type, None)
    
    def set_default(self, template_class: Type[CardItemWidget]):
        """
        Set default template for unknown types.
        
        Args:
            template_class: Fallback template class
        """
        self._default_template = template_class
    
    def set_custom_selector(
        self,
        selector: Callable[['CardItem'], Type[CardItemWidget]]
    ):
        """
        Set custom selector function.
        
        Args:
            selector: Function(item) returning template class
        """
        self._custom_selector = selector
    
    def select(self, item: 'CardItem') -> Type[CardItemWidget]:
        """
        Select template class for item.
        
        Args:
            item: CardItem to get template for
            
        Returns:
            Template class
        """
        # Try custom selector first
        if self._custom_selector:
            try:
                return self._custom_selector(item)
            except Exception as e:
                logger.warning(f"Custom selector failed: {e}")
        
        # Try registered templates
        if item.item_type in self._templates:
            return self._templates[item.item_type]
        
        # Fall back to default
        return self._default_template
    
    def create(self, item: 'CardItem') -> CardItemWidget:
        """
        Create template instance for item.
        
        Args:
            item: CardItem to create widget for
            
        Returns:
            New widget instance
        """
        template_class = self.select(item)
        widget = template_class()
        widget.bind_data(item)
        return widget
    
    @property
    def registered_types(self) -> list[str]:
        """Get list of registered item types."""
        return list(self._templates.keys())
