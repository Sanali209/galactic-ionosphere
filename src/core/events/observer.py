"""
Simple Observer/Signal pattern implementation.
"""
from typing import Callable, List, Any
import asyncio
from loguru import logger

class Signal:
    """
    Synchronous signal implementation (Observer Pattern).
    Mimics Qt's Signal API for pure Python code.
    """
    def __init__(self, name: str = "Signal"):
        self.name = name
        self._handlers: List[Callable] = []

    def connect(self, handler: Callable):
        """Connect a callback."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def disconnect(self, handler: Callable):
        """Disconnect a callback."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def emit(self, *args, **kwargs):
        """Emit the signal to all connected handlers."""
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # We can't await sync signal, so we schedule it
                    # But ideally Signal should be used for sync handlers
                    # If we have a running loop, create task
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(*args, **kwargs))
                    except RuntimeError:
                        # No loop
                        logger.warning(f"Async handler connected to sync Signal {self.name} outside loop")
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in signal handler {self.name}: {e}")

ObserverEvent = Signal  # Alias
