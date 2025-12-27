"""
EventBus - Unified Event System

Provides a single event bus for decoupled publish/subscribe communication.
Replaces multiple event patterns with one consistent API.
"""
import asyncio
import inspect
from typing import Any, Callable, Dict, List
from loguru import logger

from src.core.base_system import BaseSystem


class EventBus(BaseSystem):
    """
    Unified event bus for application-wide pub/sub.
    
    Usage:
        # Subscribe
        event_bus.subscribe("file.updated", handle_file_update)
        
        # Publish
        await event_bus.publish("file.updated", {"path": "/foo/bar"})
    """
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def initialize(self):
        """Initialize event bus."""
        logger.info("EventBus initialized")
        await super().initialize()
    
    async def shutdown(self):
        """Shutdown event bus."""
        self._subscribers.clear()
        await super().shutdown()
    
    def subscribe(self, event: str, handler: Callable) -> None:
        """
        Subscribe to an event.
        
        Args:
            event: Event name (e.g., "file.updated", "scan.complete")
            handler: Callback function (sync or async)
        """
        if event not in self._subscribers:
            self._subscribers[event] = []
        
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)
            logger.debug(f"Subscribed to {event}: {handler.__name__}")
    
    def unsubscribe(self, event: str, handler: Callable) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event: Event name
            handler: Handler to remove
        """
        if event in self._subscribers and handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)
            logger.debug(f"Unsubscribed from {event}: {handler.__name__}")
    
    async def publish(self, event: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event name
            data: Optional data to pass to handlers
        """
        handlers = self._subscribers.get(event, [])
        
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in handler for {event}: {e}")
    
    def publish_sync(self, event: str, data: Any = None) -> None:
        """
        Publish an event synchronously (for Qt signal handlers).
        
        Args:
            event: Event name
            data: Optional data to pass to handlers
        """
        handlers = self._subscribers.get(event, [])
        
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in sync handler for {event}: {e}")
