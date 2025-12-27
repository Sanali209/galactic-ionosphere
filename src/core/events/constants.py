"""
Event Type Constants.

Standard event types for application-wide pub/sub messaging.
Use these constants with EventBus for type-safe event handling.

Usage:
    from src.core.events import Events, EventBus
    
    event_bus.subscribe(Events.FILE_CREATED, on_file_created)
    await event_bus.publish(Events.FILE_CREATED, {"path": "/foo/bar.txt"})
"""


class Events:
    """
    Standard event type constants for EventBus.
    
    Provides centralized, type-safe event names to avoid string literal errors.
    Organize events by domain (file, system, config, app lifecycle).
    
    Example:
        >>> from src.core.events import Events, EventBus
        >>> event_bus.subscribe(Events.FILE_CREATED, handler)
    """
    
    # File events - filesystem changes
    FILE_CREATED = "file.created"
    FILE_DELETED = "file.deleted"
    FILE_MOVED = "file.moved"
    FILE_RENAMED = "file.renamed"
    FILE_MODIFIED = "file.modified"
    
    # System events - BaseSystem lifecycle
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    
    # Config events - configuration changes
    CONFIG_CHANGED = "config.changed"
    CONFIG_LOADED = "config.loaded"
    CONFIG_SAVED = "config.saved"
    
    # Application lifecycle events
    APP_INITIALIZED = "app.initialized"
    APP_STARTED = "app.started"
    APP_STOPPING = "app.stopping"
    APP_STOPPED = "app.stopped"
    
    # Scan/Discovery events
    SCAN_STARTED = "scan.started"
    SCAN_PROGRESS = "scan.progress"
    SCAN_COMPLETED = "scan.completed"
    SCAN_ERROR = "scan.error"
    
    # Processing events
    PROCESSING_STARTED = "processing.started"
    PROCESSING_PROGRESS = "processing.progress"
    PROCESSING_COMPLETED = "processing.completed"
    PROCESSING_ERROR = "processing.error"
