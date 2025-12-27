"""
Example Undoable Commands - Reusable command implementations.

Provides common command patterns for typical use cases:
- SetPropertyCommand: Generic property setter with undo
- CompositeCommand: Group multiple commands as one
"""
from typing import Any, List

from .base import UndoableCommand


class SetPropertyCommand(UndoableCommand):
    """
    Generic command to set a property with undo support.
    
    Captures the old value on construction for undo.
    
    Example:
        # Change file rating
        cmd = SetPropertyCommand(file_record, "rating", 5)
        undo_manager.execute(cmd)
        
        # Later: undo restores original rating
        undo_manager.undo()
    """
    
    def __init__(self, target: Any, property_name: str, new_value: Any, 
                 old_value: Any = None):
        """
        Initialize property change command.
        
        Args:
            target: Object to modify
            property_name: Name of property to change
            new_value: New value to set
            old_value: Previous value (auto-captured if None)
        """
        self.target = target
        self.property_name = property_name
        self.new_value = new_value
        
        # Capture old value if not provided
        if old_value is None:
            self.old_value = getattr(target, property_name, None)
        else:
            self.old_value = old_value
    
    @property
    def description(self) -> str:
        return f"Set {self.property_name} to {self.new_value}"
    
    def execute(self) -> None:
        setattr(self.target, self.property_name, self.new_value)
    
    def undo(self) -> None:
        setattr(self.target, self.property_name, self.old_value)


class CompositeCommand(UndoableCommand):
    """
    Groups multiple commands as a single undoable unit.
    
    All sub-commands execute together and undo together.
    Undo happens in reverse order of execution.
    
    Example:
        # Move file (rename + update path)
        commands = [
            SetPropertyCommand(file, "name", new_name),
            SetPropertyCommand(file, "path", new_path),
        ]
        composite = CompositeCommand(commands, "Move file")
        undo_manager.execute(composite)
        
        # Single undo reverts both changes
        undo_manager.undo()
    """
    
    def __init__(self, commands: List[UndoableCommand], 
                 description: str = "Composite Command"):
        """
        Initialize composite command.
        
        Args:
            commands: List of commands to execute together
            description: Description for this composite
        """
        self._commands = commands
        self._description = description
    
    @property
    def description(self) -> str:
        return self._description
    
    def execute(self) -> None:
        """Execute all sub-commands in order."""
        for cmd in self._commands:
            cmd.execute()
    
    def undo(self) -> None:
        """Undo all sub-commands in reverse order."""
        for cmd in reversed(self._commands):
            cmd.undo()
    
    def redo(self) -> None:
        """Redo all sub-commands in order."""
        for cmd in self._commands:
            cmd.redo()
