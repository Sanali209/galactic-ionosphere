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
        Initialize all registered systems in dependency order.
        
        Systems can declare dependencies using `depends_on` class attribute:
            class MyService(BaseSystem):
                depends_on = [DatabaseManager, ConfigService]
        """
        logger.info("Starting all systems...")
        
        # Build dependency order using topological sort
        ordered = self._topological_sort()
        
        for system in ordered:
            try:
                await system.initialize()
                logger.info(f"System {system.__class__.__name__} started.")
            except Exception as e:
                logger.error(f"Failed to start system {system.__class__.__name__}: {e}")
    
    def _topological_sort(self) -> list:
        """
        Sort systems by dependencies (topological order).
        
        Returns:
            List of systems in safe start order
        """
        # Build adjacency list
        in_degree = {sys: 0 for sys in self._systems.values()}
        graph = {sys: [] for sys in self._systems.values()}
        
        for sys in self._systems.values():
            deps = getattr(sys.__class__, 'depends_on', [])
            for dep_cls in deps:
                if dep_cls in self._systems:
                    dep_sys = self._systems[dep_cls]
                    graph[dep_sys].append(sys)
                    in_degree[sys] += 1
        
        # Kahn's algorithm
        queue = [sys for sys, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            sys = queue.pop(0)
            result.append(sys)
            
            for dependent in graph[sys]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(self._systems):
            logger.warning("Circular dependency detected, using registration order")
            return list(self._systems.values())
        
        return result

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
