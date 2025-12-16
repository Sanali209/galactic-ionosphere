from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict
from loguru import logger

class IDriver(ABC):
    @property
    @abstractmethod
    def id(self) -> str: pass
    
    @abstractmethod
    def load(self): pass
    
    @abstractmethod
    def unload(self): pass

T = TypeVar('T', bound=IDriver)

class DriverManager(Generic[T]):
    """
    Manages a group of interchangeable drivers (plugins).
    """
    def __init__(self, name: str):
        self.name = name
        self._drivers: Dict[str, T] = {}
        self._active: T = None

    def register(self, driver: T):
        logger.debug(f"[{self.name}] Registering driver: {driver.id}")
        self._drivers[driver.id] = driver

    def switch(self, driver_id: str):
        if driver_id not in self._drivers:
            raise ValueError(f"Driver {driver_id} missing in {self.name}")
            
        if self._active:
            if self._active.id == driver_id:
                return # Already active
            logger.info(f"[{self.name}] Unloading {self._active.id}")
            self._active.unload()
            
        new_driver = self._drivers[driver_id]
        logger.info(f"[{self.name}] Loading {new_driver.id}")
        new_driver.load()
        self._active = new_driver

    def get(self) -> T:
        if not self._active:
            raise RuntimeError(f"No active driver for {self.name}")
        return self._active
