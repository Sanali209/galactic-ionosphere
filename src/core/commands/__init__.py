"""
Foundation Command System.

Provides Command pattern infrastructure:
- ICommand: Marker interface for commands
- ICommandHandler: Async handler interface  
- UndoableCommand: Commands with undo/redo support
- CommandBus: Dispatch and pub/sub
- UndoManager: Stack-based undo/redo management
- Example commands for common patterns
"""
from .base import ICommand, ICommandHandler, UndoableCommand
from .bus import CommandBus
from .undo_manager import UndoManager
from .examples import SetPropertyCommand, CompositeCommand

__all__ = [
    # Base interfaces
    "ICommand",
    "ICommandHandler", 
    "UndoableCommand",
    # Systems
    "CommandBus",
    "UndoManager",
    # Example implementations
    "SetPropertyCommand",
    "CompositeCommand",
]
