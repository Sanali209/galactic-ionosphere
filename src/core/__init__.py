"""
Foundation Core - Application Infrastructure.

Provides core systems for building desktop applications:
- ServiceLocator: Dependency injection and system management
- BaseSystem: Abstract base for all systems
- ConfigManager: Configuration with persistence
- EventBus: Unified pub/sub messaging
- CommandBus: Command pattern with undo/redo
- LifecycleManager: Application state machine

Usage:
    from src.core import ServiceLocator, sl, EventBus, CommandBus
    
    # Initialize application
    sl.init("config.json")
    sl.register_system(MyService)
    await sl.start_all()
"""
from .base_system import BaseSystem
from .locator import ServiceLocator, sl
from .config import (
    ConfigManager, 
    AppConfig, 
    AISettings, 
    MongoSettings, 
    GeneralSettings,
    ProcessingSettings,
)
from .lifecycle import LifecycleManager, AppState, LifecycleError
from .events import ObserverEvent, EventBus, Events
from .commands import (
    ICommand, 
    ICommandHandler, 
    UndoableCommand,
    CommandBus, 
    UndoManager,
    SetPropertyCommand, 
    CompositeCommand,
)
from .decorators import system, on_lifecycle, subscribe_event

__all__ = [
    # Core infrastructure
    "BaseSystem",
    "ServiceLocator",
    "sl",
    
    # Configuration
    "ConfigManager",
    "AppConfig",
    "AISettings",
    "MongoSettings",
    "GeneralSettings",
    "ProcessingSettings",
    
    # Lifecycle
    "LifecycleManager",
    "AppState",
    "LifecycleError",
    
    # Events
    "ObserverEvent",
    "EventBus",
    "Events",
    
    # Commands
    "ICommand",
    "ICommandHandler",
    "UndoableCommand",
    "CommandBus",
    "UndoManager",
    "SetPropertyCommand",
    "CompositeCommand",
    
    # Decorators
    "system",
    "on_lifecycle",
    "subscribe_event",
]
