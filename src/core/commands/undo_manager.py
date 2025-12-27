"""
Undo Manager - Stack-based undo/redo management.

Provides UndoManager system for executing UndoableCommands with full
undo/redo support and Qt signals for UI binding.
"""
from typing import List, Optional
from loguru import logger
from PySide6.QtCore import QObject, Signal

from .base import UndoableCommand
from ..base_system import BaseSystem
from ..service_decorator import Service


class _UndoSignals(QObject):
    """Signal holder to avoid metaclass conflict."""
    can_undo_changed = Signal(bool)
    can_redo_changed = Signal(bool)
    state_changed = Signal()


@Service
class UndoManager(BaseSystem):
    """
    Manages undo/redo stacks for UndoableCommands.
    
    Features:
    - Configurable max history size
    - Qt signals for UI binding (via signals property)
    - Clear history
    - Get undo/redo descriptions for UI
    
    Usage:
        # Get from ServiceLocator
        undo_mgr = locator.get_system(UndoManager)
        
        # Execute undoable command
        cmd = RenameFileCommand(file, "old.txt", "new.txt")
        undo_mgr.execute(cmd)
        
        # Undo/Redo
        undo_mgr.undo()  # Reverts to old.txt
        undo_mgr.redo()  # Back to new.txt
        
        # Connect UI via signals property
        undo_action.setEnabled(undo_mgr.can_undo)
        undo_mgr.signals.can_undo_changed.connect(undo_action.setEnabled)
    """
    
    def __init__(self, locator, config, max_history: int = 100):
        """
        Initialize UndoManager.
        
        Args:
            locator: ServiceLocator instance
            config: ConfigManager instance
            max_history: Maximum commands to keep in undo stack
        """
        super().__init__(locator, config)
        
        # Qt signals via composition
        self._signals = _UndoSignals()
        
        self._undo_stack: List[UndoableCommand] = []
        self._redo_stack: List[UndoableCommand] = []
        self._max_history = max_history
        
        # Track previous state for signal emission
        self._last_can_undo = False
        self._last_can_redo = False
    
    @property
    def signals(self) -> _UndoSignals:
        """Get Qt signals object for UI binding."""
        return self._signals
    
    async def initialize(self):
        """Initialize the UndoManager."""
        await super().initialize()
    
    async def shutdown(self):
        """Shutdown and clear stacks."""
        self.clear()
        await super().shutdown()
    
    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0
    
    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    @property
    def undo_description(self) -> Optional[str]:
        """Get description of next undo action."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return None
    
    @property
    def redo_description(self) -> Optional[str]:
        """Get description of next redo action."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return None
    
    def execute(self, command: UndoableCommand) -> None:
        """
        Execute a command and add to undo stack.
        
        Args:
            command: UndoableCommand to execute
        """
        try:
            command.execute()
            
            # Add to undo stack
            self._undo_stack.append(command)
            
            # Enforce max history
            while len(self._undo_stack) > self._max_history:
                self._undo_stack.pop(0)
            
            # Clear redo stack (new action breaks redo chain)
            self._redo_stack.clear()
            
            logger.debug(f"Executed: {command.description}")
            self._emit_state_changes()
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def undo(self) -> bool:
        """
        Undo the last command.
        
        Returns:
            True if undo was performed, False if nothing to undo
        """
        if not self._undo_stack:
            return False
        
        command = self._undo_stack.pop()
        
        try:
            command.undo()
            self._redo_stack.append(command)
            
            logger.debug(f"Undone: {command.description}")
            self._emit_state_changes()
            return True
            
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            # Put it back on undo stack
            self._undo_stack.append(command)
            raise
    
    def redo(self) -> bool:
        """
        Redo the last undone command.
        
        Returns:
            True if redo was performed, False if nothing to redo
        """
        if not self._redo_stack:
            return False
        
        command = self._redo_stack.pop()
        
        try:
            command.redo()
            self._undo_stack.append(command)
            
            logger.debug(f"Redone: {command.description}")
            self._emit_state_changes()
            return True
            
        except Exception as e:
            logger.error(f"Redo failed: {e}")
            # Put it back on redo stack
            self._redo_stack.append(command)
            raise
    
    def clear(self) -> None:
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.debug("Undo history cleared")
        self._emit_state_changes()
    
    def _emit_state_changes(self) -> None:
        """Emit signals if can_undo/can_redo state changed."""
        current_can_undo = self.can_undo
        current_can_redo = self.can_redo
        
        if current_can_undo != self._last_can_undo:
            self._last_can_undo = current_can_undo
            self._signals.can_undo_changed.emit(current_can_undo)
        
        if current_can_redo != self._last_can_redo:
            self._last_can_redo = current_can_redo
            self._signals.can_redo_changed.emit(current_can_redo)
        
        self._signals.state_changed.emit()

