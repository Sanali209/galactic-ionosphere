"""
Foundation Command Pattern - Base Interfaces.

Provides:
- ICommand: Marker interface for all commands
- ICommandHandler: Async handler interface
- UndoableCommand: Command with undo/redo support
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic


class ICommand(ABC):
    """Marker interface for commands dispatched via CommandBus."""
    pass


C = TypeVar('C', bound=ICommand)


class ICommandHandler(Generic[C], ABC):
    """
    Async handler for a specific command type.
    
    Example:
        class CreateFileHandler(ICommandHandler[CreateFileCommand]):
            async def handle(self, command: CreateFileCommand):
                # Handle the command
                pass
    """
    @abstractmethod
    async def handle(self, command: C):
        """Handle the command asynchronously."""
        pass


class UndoableCommand(ICommand, ABC):
    """
    Command that supports undo/redo operations.
    
    Use this for operations that modify state and should be reversible.
    Execute via UndoManager to enable undo/redo functionality.
    
    Example:
        class RenameFileCommand(UndoableCommand):
            def __init__(self, file, old_name, new_name):
                self.file = file
                self.old_name = old_name
                self.new_name = new_name
            
            @property
            def description(self) -> str:
                return f"Rename to {self.new_name}"
            
            def execute(self):
                self.file.name = self.new_name
            
            def undo(self):
                self.file.name = self.old_name
    """
    
    @property
    def description(self) -> str:
        """
        Human-readable description for UI display.
        
        Returns:
            Description string (default: class name)
        """
        return self.__class__.__name__
    
    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command (forward operation).
        
        This is called when the command is first run and on redo.
        """
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """
        Reverse the command.
        
        Must restore state to exactly what it was before execute().
        """
        pass
    
    def redo(self) -> None:
        """
        Re-execute the command after undo.
        
        Default implementation calls execute().
        Override if redo requires different logic.
        """
        self.execute()

