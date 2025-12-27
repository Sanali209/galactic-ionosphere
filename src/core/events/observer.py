from loguru import logger
from typing import Callable, List, Any

class Signal:
    """
    A simple observer pattern implementation (Synchronous).
    Allows subscribers to connect to this signal and receive notifications.
    Equivalent to Qt's Signal or C#'s event.
    """
    def __init__(self, name: str = "Signal"):
        self.name = name
        self._subscribers: List[Callable] = []

    def connect(self, callback: Callable):
        """Connect a callback function to this signal."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def disconnect(self, callback: Callable):
        """Disconnect a callback function from this signal."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, *args, **kwargs):
        """Broadcast arguments to all subscribers synchronously."""
        for sub in self._subscribers:
            try:
                sub(*args, **kwargs)
            except Exception as e:
                logger.error(f"Signal '{self.name}' error in subscriber '{sub}': {e}")

class ObserverEvent(Signal):
    """
    DEPRECATED: Use `Signal` instead.
    Maintained for backward compatibility.
    """
    def __init__(self, name: str):
        import warnings
        warnings.warn("ObserverEvent is deprecated, use Signal instead", DeprecationWarning, stacklevel=2)
        super().__init__(name)

