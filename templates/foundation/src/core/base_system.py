from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locator import ServiceLocator
    from .config import ConfigManager

class BaseSystem(ABC):
    """
    Abstract Base Class for all core systems (Assets, Journal, Tasks, etc.).
    Ensures consistent initialization and access to globals (Locator, Config).
    """
    def __init__(self, locator: 'ServiceLocator', config: 'ConfigManager'):
        self.locator = locator
        self.config = config
        self._is_ready = False

    @abstractmethod
    async def initialize(self):
        """
        Async initialization logic (e.g. database connections, loading cache).
        Should be called by the ServiceLocator during startup.
        """
        self._is_ready = True

    @abstractmethod
    async def shutdown(self):
        """
        Cleanup logic (e.g. closing connections).
        """
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready
