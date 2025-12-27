"""
WidgetPool - Widget recycling for virtualization.

Manages a pool of reusable widgets to handle large datasets
without creating thousands of widget instances.
"""
from typing import Callable, TypeVar, Generic
from loguru import logger
from PySide6.QtWidgets import QWidget


T = TypeVar('T', bound=QWidget)


class WidgetPool(Generic[T]):
    """
    Recycles QWidget instances for virtualization.
    
    Only creates a limited number of widgets (pool_size) regardless
    of total data size. Widgets are recycled as items scroll in/out
    of the visible viewport.
    
    Features:
    - Pre-allocated widget pool
    - Acquire/release pattern
    - Automatic cleanup of out-of-viewport widgets
    - Template factory for creating new widgets
    
    Example:
        pool = WidgetPool(
            factory=lambda: CardItemWidget(),
            pool_size=50
        )
        
        # When item becomes visible
        widget = pool.acquire("item_123")
        widget.bind_data(item_data)
        
        # When item leaves viewport
        pool.release("item_123")
    """
    
    def __init__(
        self, 
        factory: Callable[[], T],
        pool_size: int = 50,
        prealloc: bool = False
    ):
        """
        Initialize widget pool.
        
        Args:
            factory: Function that creates new widget instances
            pool_size: Maximum number of widgets to create
            prealloc: If True, pre-allocate all widgets on init
        """
        self._factory = factory
        self._pool_size = pool_size
        self._free_widgets: list[T] = []
        self._active_widgets: dict[str, T] = {}  # item_id → widget
        
        if prealloc:
            self._preallocate()
    
    def _preallocate(self):
        """Pre-allocate pool_size widgets."""
        logger.debug(f"Pre-allocating {self._pool_size} widgets")
        for _ in range(self._pool_size):
            widget = self._factory()
            widget.hide()
            self._free_widgets.append(widget)
    
    @property
    def active_count(self) -> int:
        """Number of widgets currently in use."""
        return len(self._active_widgets)
    
    @property
    def free_count(self) -> int:
        """Number of widgets available for reuse."""
        return len(self._free_widgets)
    
    @property
    def total_count(self) -> int:
        """Total number of widgets created."""
        return self.active_count + self.free_count
    
    @property
    def active_ids(self) -> set[str]:
        """Set of item IDs with active widgets."""
        return set(self._active_widgets.keys())
    
    def acquire(self, item_id: str) -> T:
        """
        Get widget for item (recycled or new).
        
        Args:
            item_id: Unique identifier for the item
            
        Returns:
            Widget instance (recycled or newly created)
        """
        # Check if already active
        if item_id in self._active_widgets:
            return self._active_widgets[item_id]
        
        # Try to get from free pool
        if self._free_widgets:
            widget = self._free_widgets.pop()
            logger.trace(f"Recycled widget for {item_id}")
        else:
            # Create new widget (up to pool_size)
            if self.total_count < self._pool_size:
                widget = self._factory()
                logger.trace(f"Created new widget for {item_id}")
            else:
                # Pool exhausted - should not happen with proper viewport calc
                logger.warning(
                    f"Widget pool exhausted ({self._pool_size}). "
                    "Consider increasing pool_size."
                )
                # Force recycle oldest active widget
                oldest_id = next(iter(self._active_widgets))
                widget = self._active_widgets.pop(oldest_id)
                self._reset_widget(widget)
                logger.debug(f"Force-recycled widget from {oldest_id}")
        
        self._active_widgets[item_id] = widget
        widget.show()
        return widget
    
    def release(self, item_id: str) -> bool:
        """
        Return widget to pool for recycling.
        
        Args:
            item_id: ID of item to release
            
        Returns:
            True if widget was released, False if not found
        """
        widget = self._active_widgets.pop(item_id, None)
        if widget:
            self._reset_widget(widget)
            widget.hide()
            self._free_widgets.append(widget)
            logger.trace(f"Released widget for {item_id}")
            return True
        return False
    
    def release_outside_viewport(self, visible_ids: set[str]) -> int:
        """
        Release all widgets not in visible set.
        
        Args:
            visible_ids: Set of item IDs that should remain active
            
        Returns:
            Number of widgets released
        """
        released = 0
        for item_id in list(self._active_widgets.keys()):
            if item_id not in visible_ids:
                self.release(item_id)
                released += 1
        
        if released > 0:
            logger.debug(f"Released {released} out-of-viewport widgets")
        
        return released
    
    def get_widget(self, item_id: str) -> T | None:
        """
        Get active widget for item if exists.
        
        Args:
            item_id: Item identifier
            
        Returns:
            Widget if active, None otherwise
        """
        return self._active_widgets.get(item_id)
    
    def is_active(self, item_id: str) -> bool:
        """Check if item has active widget."""
        return item_id in self._active_widgets
    
    def _reset_widget(self, widget: T):
        """
        Reset widget state for recycling.
        
        Calls widget.reset() if available, otherwise does nothing.
        """
        if hasattr(widget, 'reset'):
            widget.reset()
    
    def clear(self):
        """Release all widgets and clear pool."""
        for widget in self._active_widgets.values():
            self._reset_widget(widget)
            widget.hide()
        
        self._free_widgets.extend(self._active_widgets.values())
        self._active_widgets.clear()
        logger.debug("Cleared all active widgets")
    
    def dispose(self):
        """Dispose all widgets and cleanup."""
        for widget in self._active_widgets.values():
            widget.deleteLater()
        for widget in self._free_widgets:
            widget.deleteLater()
        
        self._active_widgets.clear()
        self._free_widgets.clear()
        logger.debug("Disposed widget pool")
