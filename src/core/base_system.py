from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .locator import ServiceLocator
    from .config import ConfigManager

class BaseSystem(ABC):
    """
    Abstract Base Class for all core systems (Assets, Journal, Tasks, etc.).
    Ensures consistent initialization and access to globals (Locator, Config).
    
    Supports automatic event subscription via @subscribe_event decorator:
        from src.core.decorators import subscribe_event
        
        class MyService(BaseSystem):
            @subscribe_event("file.created", "file.updated")
            async def on_file_event(self, data):
                # Handle event
                pass
    """
    def __init__(self, locator: 'ServiceLocator', config: 'ConfigManager'):
        self.locator = locator
        self.config = config
        self._is_ready = False

    @abstractmethod
    async def initialize(self):
        """
        Async initialization logic (e.g. database connections, loading cache).
        Should be called by the ServiceLocator during startup.
        
        Automatically subscribes methods decorated with @subscribe_event.
        """
        # Auto-subscribe decorated event handlers
        self._auto_subscribe_events()
        self._is_ready = True

    def _auto_subscribe_events(self) -> None:
        """
        Scan for methods decorated with @subscribe_event and subscribe them.
        
        Methods decorated with @subscribe_event("event.type") will have
        a _subscribed_events attribute containing event types to subscribe to.
        """
        from .events import EventBus
        
        # Get EventBus from locator
        try:
            bus = self.locator.get_system(EventBus)
        except KeyError:
            logger.warning(f"{self.__class__.__name__}: EventBus not available for auto-subscription")
            return
        
        # Scan all methods for _subscribed_events attribute
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_subscribed_events'):
                events = method._subscribed_events
                for event in events:
                    bus.subscribe(event, method)
                    logger.debug(f"{self.__class__.__name__}.{name} auto-subscribed to: {event}")

    @abstractmethod
    async def shutdown(self):
        """
        Cleanup logic (e.g. closing connections).
        """
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def __aenter__(self):
        """Async context manager entry: Initialize system."""
        if not self._is_ready:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit: Shutdown system."""
        if self._is_ready:
            await self.shutdown()

