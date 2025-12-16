from typing import Type, TypeVar, Dict, Optional
import asyncio
from loguru import logger

from .config import ConfigManager
from .events import ObserverEvent
from .base_system import BaseSystem

T = TypeVar('T', bound=BaseSystem)

class ServiceLocator:
    """
    Central registry for application services (Systems).
    Manages initialization and shutdown order.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceLocator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init(self, config_path: str = "config.json"):
        if getattr(self, '_initialized', False):
            return

        self.bus = ObserverEvent("SystemBus")
        self.config = ConfigManager(config_path)
        self._systems: Dict[Type[BaseSystem], BaseSystem] = {}
        
        # Reactive binding: Config -> Systems?
        # Systems subscribe to config changes themselves usually.
        
        self._initialized = True
        logger.info("ServiceLocator initialized.")

    def register_system(self, system_cls: Type[T]) -> T:
        """
        Instantiates and registers a system.
        """
        if system_cls in self._systems:
             return self._systems[system_cls]
        
        logger.debug(f"Registering system: {system_cls.__name__}")
        instance = system_cls(self, self.config)
        self._systems[system_cls] = instance
        return instance

    def get_system(self, system_cls: Type[T]) -> T:
        """
        Retrieves a registered system.
        """
        if system_cls not in self._systems:
            raise KeyError(f"System {system_cls.__name__} not registered.")
        return self._systems[system_cls]

    async def start_all(self):
        """
        Initialize all registered systems.
        """
        logger.info("Starting all systems...")
        for system in self._systems.values():
            try:
                await system.initialize()
                logger.info(f"System {system.__class__.__name__} started.")
            except Exception as e:
                logger.error(f"Failed to start system {system.__class__.__name__}: {e}")

    async def stop_all(self):
        """
        Shutdown all systems in reverse order.
        """
        logger.info("Stopping all systems...")
        # Reverse order often better for dependencies
        systems = list(self._systems.values())
        for system in reversed(systems):
            try:
                await system.shutdown()
                logger.info(f"System {system.__class__.__name__} stopped.")
            except Exception as e:
                logger.error(f"Failed to stop system {system.__class__.__name__}: {e}")

# Global access
sl = ServiceLocator()
