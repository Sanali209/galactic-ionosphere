from typing import Dict, Type
from loguru import logger
from .base import ICommand, ICommandHandler
from ..base_system import BaseSystem

class CommandBus(BaseSystem):
    """
    Dispatches commands to registered handlers.
    """
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._handlers: Dict[Type[ICommand], ICommandHandler] = {}

    async def initialize(self):
        logger.info("CommandBus initialized.")
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()

    def register(self, command_cls: Type[ICommand], handler: ICommandHandler):
        self._handlers[command_cls] = handler

    async def dispatch(self, command: ICommand):
        cmd_type = type(command)
        if cmd_type not in self._handlers:
            raise ValueError(f"No handler registered for {cmd_type.__name__}")
        
        handler = self._handlers[cmd_type]
        try:
            await handler.handle(command)
        except Exception as e:
            logger.error(f"Error handling command {cmd_type.__name__}: {e}")
            raise
