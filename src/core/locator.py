from typing import Type, TypeVar, Dict, Optional
import asyncio
from loguru import logger

from .config import ConfigManager
from .events import EventBus
from .base_system import BaseSystem

T = TypeVar('T', bound=BaseSystem)

class ServiceLocator:
    """
    Central registry for application services (Systems).
    Manages initialization and shutdown order.
    """
    # Remove Singleton pattern to allow multiple instances (Main vs Engine)
        # _instance = None 
        # def __new__(cls): ... REMOVED

    def __init__(self):
        """Initialize empty ServiceLocator. Call init() to fully configure."""
        self._systems: Dict[Type[BaseSystem], BaseSystem] = {}
        self._startup_order: list = []
        self._initialized = False
        self.config = None
        self.bus = None

    def init(self, config_path: str = "config.json"):
        if getattr(self, '_initialized', False):
            return

        self.config = ConfigManager(config_path)
        self._systems: Dict[Type[BaseSystem], BaseSystem] = {}
        # Register ConfigManager explicitly
        self._systems[ConfigManager] = self.config
        
        self._startup_order: list = []  # Track actual startup order for safe shutdown
        
        # Register EventBus as a core system (initialized in start_all with other systems)
        self.bus = EventBus(self, self.config)
        self._systems[EventBus] = self.bus
        # NOTE: EventBus.initialize() is async - called by start_all() in proper order
        
        # Reactive binding: Config -> Systems?
        # Systems subscribe to config changes themselves usually.
        
        self._initialized = True
        logger.info("ServiceLocator initialized.")

    def register_system(self, system_cls: Type[T]) -> T:
        """
        Instantiates and registers a system.
        """
        if system_cls in self._systems:
            logger.warning(f"Register: System {system_cls.__name__} (id={id(system_cls)}) ALREADY registered. Returning existing id={id(self._systems[system_cls])}")
            return self._systems[system_cls]
        
        logger.debug(f"Registering system: {system_cls.__name__} (id={id(system_cls)})")
        instance = system_cls(self, self.config)
        logger.debug(f"Created instance: {system_cls.__name__} (id={id(instance)})")
        self._systems[system_cls] = instance
        return instance

    def register_instance(self, system_cls: Type[T], instance: T) -> T:
        """
        Registers an existing system instance.
        Useful for UI-bound services created after bootstrap.
        """
        if system_cls in self._systems:
            logger.warning(f"System {system_cls.__name__} already registered. Overwriting.")
        
        self._systems[system_cls] = instance
        if instance not in self._startup_order:
            self._startup_order.append(instance)
        logger.info(f"Instance registered: {system_cls.__name__}")
        return instance

    def get_system(self, system_cls: Type[T]) -> T:
        """
        Retrieves a registered system.
        """
        if system_cls not in self._systems:
            raise KeyError(f"System {system_cls.__name__} not registered.")
        return self._systems[system_cls]

    def has_system(self, system_cls: Type[BaseSystem]) -> bool:
        """
        Check if a system is registered.
        """
        return system_cls in self._systems

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
        self._startup_order = ordered  # Store for safe shutdown
        
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
        Shutdown all systems in reverse startup order.
        
        Uses the actual startup order (respecting dependencies) rather than
        registration order for safer shutdown.
        """
        logger.info("Stopping all systems...")
        # Use reverse of actual startup order for safe shutdown
        systems = reversed(self._startup_order) if self._startup_order else reversed(list(self._systems.values()))
        for system in systems:
            try:
                await system.shutdown()
                logger.info(f"System {system.__class__.__name__} stopped.")
            except Exception as e:
                logger.error(f"Failed to stop system {system.__class__.__name__}: {e}")

# Context-aware Global Access
from contextvars import ContextVar
import sys

# Create the default Main Thread locator
_main_sl = ServiceLocator()

# ContextVar holding the current active locator (defaults to Main)
_current_sl = ContextVar("current_service_locator", default=_main_sl)

class ServiceLocatorProxy:
    """Proxy that delegates to the context-local ServiceLocator."""
    def __getattr__(self, name):
        return getattr(_current_sl.get(), name)
        
    def __repr__(self):
        return repr(_current_sl.get())
        
# Export the proxy as 'sl'
sl = ServiceLocatorProxy()

def set_active_locator(locator: ServiceLocator):
    """Set the active ServiceLocator for the current context (Thread/Task)."""
    _current_sl.set(locator)
    
def get_active_locator() -> ServiceLocator:
    """Get the active ServiceLocator."""
    return _current_sl.get()
