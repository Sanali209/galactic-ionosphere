from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict

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
    """Manages the lifecycle of drivers of one group"""
    def __init__(self, name: str):
        self.name = name
        self._drivers: Dict[str, T] = {}
        self._active: T = None

    def register(self, driver: T):
        self._drivers[driver.id] = driver

    def switch(self, driver_id: str):
        if driver_id not in self._drivers:
            raise ValueError(f"Driver {driver_id} missing in {self.name}")
            
        if self._active:
            self._active.unload()
            
        new_driver = self._drivers[driver_id]
        new_driver.load()
        self._active = new_driver

    def get(self) -> T:
        if not self._active:
            raise RuntimeError(f"No active driver for {self.name}")
        return self._active

class CoreFacade:
    def __init__(self):
        # Defined generic driver managers
        # In a real app we'd likely have specific interfaces for these generic types
        # e.g. IVectorDriver, ILlmDriver, IStorageDriver
        # For now, we use IDriver as the bound
        self.ai_vectors = DriverManager[IDriver]("Vectors")
        self.ai_llm = DriverManager[IDriver]("LLM")
        self.storage = DriverManager[IDriver]("Storage")
