from typing import Dict, Type
from loguru import logger
from .base import ICommand, ICommandHandler, UndoableCommand
from ..base_system import BaseSystem


class CommandBus(BaseSystem):
    """
    Dispatches commands to registered handlers.
    
    Provides:
    - Traditional async dispatch with handlers (Command Pattern)
    - Undoable command execution via UndoManager
    
    For pub/sub messaging, use EventBus from src.core.events instead.
    
    Example:
        # Register handler
        command_bus.register(CreateFileCommand, CreateFileHandler())
        
        # Dispatch command
        await command_bus.dispatch(CreateFileCommand(path="/foo.txt"))
        
        # Execute undoable command
        command_bus.dispatch_undoable(RenameCommand(old="a.txt", new="b.txt"))
    """
    
    def __init__(self, locator, config):
        """
        Initialize CommandBus.
        
        Args:
            locator: ServiceLocator instance
            config: ConfigManager instance
        """
        super().__init__(locator, config)
        self._handlers: Dict[Type[ICommand], ICommandHandler] = {}

    async def initialize(self) -> None:
        """Initialize the command bus."""
        logger.info("CommandBus initialized.")
        await super().initialize()

    async def shutdown(self) -> None:
        """Shutdown the command bus."""
        self._handlers.clear()
        await super().shutdown()

    def register(self, command_cls: Type[ICommand], handler: ICommandHandler) -> None:
        """
        Register a handler for a command type.
        
        Args:
            command_cls: Command class to handle
            handler: Handler instance implementing ICommandHandler
        """
        self._handlers[command_cls] = handler
        logger.debug(f"Registered handler for {command_cls.__name__}")

    async def dispatch(self, command: ICommand) -> None:
        """
        Dispatch command to its registered handler.
        
        Args:
            command: Command instance to dispatch
            
        Raises:
            ValueError: If no handler is registered for the command type
        """
        cmd_type = type(command)
        if cmd_type not in self._handlers:
            raise ValueError(f"No handler registered for {cmd_type.__name__}")
        
        handler = self._handlers[cmd_type]
        try:
            await handler.handle(command)
            logger.debug(f"Command {cmd_type.__name__} handled successfully")
        except Exception as e:
            logger.error(f"Error handling command {cmd_type.__name__}: {e}")
            raise
    
    def dispatch_undoable(self, command: UndoableCommand) -> bool:
        """
        Execute an undoable command via UndoManager.
        
        If UndoManager is registered, the command is executed through it
        to enable undo/redo. Otherwise, executes directly.
        
        Args:
            command: UndoableCommand to execute
            
        Returns:
            True if command executed successfully, False otherwise
        """
        try:
            from .undo_manager import UndoManager
            undo_mgr = self.locator.get_system(UndoManager)
            undo_mgr.execute(command)
            return True
        except KeyError:
            # UndoManager not registered, execute directly
            logger.debug("UndoManager not registered, executing command directly")
            try:
                command.execute()
                return True
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Undoable command dispatch failed: {e}")
            return False


