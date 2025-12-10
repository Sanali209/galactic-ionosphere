import logging
from typing import Callable, List, Any

class ObserverEvent:
    def __init__(self, name: str):
        self.name = name
        self._subscribers: List[Callable] = []

    def connect(self, callback: Callable):
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def disconnect(self, callback: Callable):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, *args, **kwargs):
        for sub in self._subscribers:
            try:
                sub(*args, **kwargs)
            except Exception as e:
                logging.error(f"Event '{self.name}' error in subscriber '{sub}': {e}")
