"""
Event System - Unified Pub/Sub Messaging.

Provides:
- ObserverEvent: Simple observer pattern for sync notifications (e.g., config changes)
- EventBus: Unified pub/sub for async application-wide events (recommended for cross-system messaging)
- Events: Standard event type constants for type-safe subscriptions

Usage:
    from src.core.events import EventBus, Events
    
    # Subscribe to events
    event_bus.subscribe(Events.FILE_CREATED, on_file_created)
    
    # Publish events
    await event_bus.publish(Events.FILE_CREATED, {"path": "/foo/bar.txt"})
"""
from .observer import ObserverEvent, Signal
from .bus import EventBus
from .constants import Events


__all__ = ["Signal", "ObserverEvent", "EventBus", "Events"]
