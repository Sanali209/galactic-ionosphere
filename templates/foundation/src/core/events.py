import logging
from typing import Callable, List, Any

class ObserverEvent:
    """
    A simple observer pattern implementation.
    Allows subscribers to connect to named events and receive notifications.
    """
    def __init__(self, name: str):
        self.name = name
        self._subscribers: List[Callable] = []

    def connect(self, callback: Callable):
        """Connect a callback function to this event."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def disconnect(self, callback: Callable):
        """Disconnect a callback function from this event."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, *args, **kwargs):
        """Broadcast arguments to all subscribers."""
        for sub in self._subscribers:
            try:
                sub(*args, **kwargs)
            except Exception as e:
                logging.error(f"Event '{self.name}' error in subscriber '{sub}': {e}")
