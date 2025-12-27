"""
Backend Provider & Driver System
Modular backend registry supporting dynamic driver management, versioning, 
compatibility checks, lazy/forced loading, error handling, and rollback strategies.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Callable, Union, cast
from abc import ABC, abstractmethod
import threading
import time
from loguru import logger

from SLM.core.component import Component
from SLM.core.message_bus import MessageBus

T = TypeVar('T', bound='Driver')


class Driver(ABC):
    """
    Base class for backend drivers that encapsulates a backend API and its concrete realizations.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the driver
        
        Args:
            name: Driver name (optional, defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.version: str = "1.0"
        self.compatible_versions: List[str] = ["1.0"]
        self.metadata: Dict[str, Any] = {}
        self._loaded: bool = False
        self._realizations: Dict[str, 'Driver'] = {}
        self._active_realization: Optional[str] = None
        self._lock = threading.Lock()
        
        # Will be injected by DI system
        self.app: Optional[Any] = None
        self.message_bus: Optional[MessageBus] = None
        self.config: Optional[Any] = None

    def get_all_realizations(self) -> List['Driver']:
        """
        Get all registered realizations for this driver
        
        Returns:
            List of driver realizations
        """
        with self._lock:
            return list(self._realizations.values())

    def get_realization_by_name(self, name: str) -> Optional['Driver']:
        """
        Get a realization by name
        
        Args:
            name: Realization name
            
        Returns:
            Driver realization or None if not found
        """
        with self._lock:
            return self._realizations.get(name)

    def get_realization_by_tag(self, tag: str) -> List['Driver']:
        """
        Get realizations by tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching realizations
        """
        with self._lock:
            matching = []
            for realization in self._realizations.values():
                if 'tags' in realization.metadata and tag in realization.metadata['tags']:
                    matching.append(realization)
            return matching

    def get_realization_by_type(self, type_: Type) -> List['Driver']:
        """
        Get realizations by type
        
        Args:
            type_: Type to search for
            
        Returns:
            List of matching realizations
        """
        with self._lock:
            return [r for r in self._realizations.values() if isinstance(r, type_)]

    def get_realization_by_version(self, version: str) -> List['Driver']:
        """
        Get realizations by version
        
        Args:
            version: Version to search for
            
        Returns:
            List of matching realizations
        """
        with self._lock:
            return [r for r in self._realizations.values() if r.version == version]

    def get_realization_by_predicate(self, predicate: Callable[['Driver'], bool]) -> List['Driver']:
        """
        Get realizations by custom predicate
        
        Args:
            predicate: Function that returns True for matching realizations
            
        Returns:
            List of matching realizations
        """
        with self._lock:
            return [r for r in self._realizations.values() if predicate(r)]

    def register_realization(self, realization: 'Driver') -> None:
        """
        Register a realization for this driver
        
        Args:
            realization: Driver realization to register
        """
        with self._lock:
            if not isinstance(realization, Driver):
                raise TypeError("Realization must inherit from Driver")
            
            self._realizations[realization.name] = realization
            logger.debug(f"Registered realization {realization.name} for driver {self.name}")

    def is_compatible_with(self, version: str) -> bool:
        """
        Check if this driver is compatible with a specific version
        
        Args:
            version: Version to check compatibility with
            
        Returns:
            True if compatible
        """
        return version in self.compatible_versions

    def load(self, force: bool = False) -> None:
        """
        Load the driver (lazy or forced)
        
        Args:
            force: Force reload even if already loaded
        """
        if self._loaded and not force:
            return
            
        with self._lock:
            try:
                self.on_load()
                self._loaded = True
                logger.debug(f"Driver {self.name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading driver {self.name}: {e}")
                self.handle_load_error(e)
                raise

    def handle_load_error(self, error: Exception) -> None:
        """
        Handle loading errors and report via event bus
        
        Args:
            error: Exception that occurred during loading
        """
        error_data = {
            'driver_name': self.name,
            'driver_type': self.__class__.__name__,
            'error': str(error),
            'timestamp': time.time()
        }
        
        if self.message_bus:
            try:
                self.message_bus.publish('driver.load_error', **error_data)
            except Exception as pub_error:
                logger.error(f"Failed to publish load error for {self.name}: {pub_error}")
        
        logger.error(f"Driver {self.name} load error: {error}")

    def rollback_to(self, realization_name: str) -> None:
        """
        Rollback to a previous realization
        
        Args:
            realization_name: Name of realization to rollback to
        """
        with self._lock:
            if realization_name not in self._realizations:
                raise ValueError(f"Realization {realization_name} not found in driver {self.name}")
            
            try:
                # Load the target realization
                target_realization = self._realizations[realization_name]
                if not target_realization._loaded:
                    target_realization.load()
                
                self._active_realization = realization_name
                logger.info(f"Driver {self.name} rolled back to realization {realization_name}")
                
                # Publish rollback event
                if self.message_bus:
                    rollback_data = {
                        'driver_name': self.name,
                        'realization_name': realization_name,
                        'timestamp': time.time()
                    }
                    self.message_bus.publish('driver.rollback', **rollback_data)
                        
            except Exception as e:
                logger.error(f"Error rolling back driver {self.name} to {realization_name}: {e}")
                self.handle_load_error(e)
                raise

    def on_load(self) -> None:
        """
        Called when driver is loaded. Override in subclasses.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if driver is loaded"""
        return self._loaded

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, version={self.version}, loaded={self._loaded})"


class BackendProvider(Component):
    """
    Central registry for backend drivers, exposed via dependency injection as a singleton.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the backend provider
        
        Args:
            name: Component name (optional)
        """
        super().__init__(name)
        self._drivers: Dict[str, Driver] = {}
        self._drivers_by_type: Dict[Type, List[Driver]] = {}
        self._lock = threading.Lock()
        
        # Will be injected by DI system
        self.app: Optional[Any] = None

    def register_driver(self, driver: Driver) -> None:
        """
        Register a driver
        
        Args:
            driver: Driver instance to register
        """
        if not isinstance(driver, Driver):
            raise TypeError("Driver must inherit from Driver base class")
            
        with self._lock:
            if driver.name in self._drivers:
                logger.warning(f"Driver {driver.name} already registered, replacing...")
            
            self._drivers[driver.name] = driver
            
            # Register by type
            driver_type = type(driver)
            if driver_type not in self._drivers_by_type:
                self._drivers_by_type[driver_type] = []
            
            # Remove existing instance of same type if any
            self._drivers_by_type[driver_type] = [d for d in self._drivers_by_type[driver_type] if d.name != driver.name]
            self._drivers_by_type[driver_type].append(driver)
            
            # Inject dependencies
            if hasattr(self, 'app'):
                driver.app = self.app
            if hasattr(self, 'message_bus'):
                driver.message_bus = self.message_bus
            if hasattr(self, 'config'):
                driver.config = self.config
            
            logger.info(f"Registered driver: {driver.name} ({driver.__class__.__name__})")

    def get_all_drivers(self) -> List[Driver]:
        """
        Get all registered drivers
        
        Returns:
            List of all drivers
        """
        with self._lock:
            return list(self._drivers.values())

    def get_driver_by_name(self, name: str) -> Optional[Driver]:
        """
        Get a driver by name
        
        Args:
            name: Driver name
            
        Returns:
            Driver instance or None if not found
        """
        with self._lock:
            return self._drivers.get(name)

    def get_driver_by_type(self, type_: Type[T]) -> Optional[T]:
        """
        Get a driver by type
        
        Args:
            type_: Driver type
            
        Returns:
            Driver instance or None if not found
        """
        with self._lock:
            drivers = self._drivers_by_type.get(type_, [])
            return cast(T, drivers[0]) if drivers else None

    def get_drivers_by_type(self, type_: Type[T]) -> List[T]:
        """
        Get all drivers of a specific type
        
        Args:
            type_: Driver type
            
        Returns:
            List of matching drivers
        """
        with self._lock:
            return cast(List[T], self._drivers_by_type.get(type_, []).copy())

    def has_driver(self, name: str) -> bool:
        """
        Check if a driver is registered
        
        Args:
            name: Driver name
            
        Returns:
            True if driver is registered
        """
        with self._lock:
            return name in self._drivers

    def unregister_driver(self, name: str) -> None:
        """
        Unregister a driver
        
        Args:
            name: Driver name
        """
        with self._lock:
            if name in self._drivers:
                driver = self._drivers[name]
                del self._drivers[name]
                
                # Remove from type registry
                driver_type = type(driver)
                if driver_type in self._drivers_by_type:
                    self._drivers_by_type[driver_type] = [d for d in self._drivers_by_type[driver_type] if d.name != name]
                
                logger.info(f"Unregistered driver: {name}")

    def get_driver_count(self) -> int:
        """
        Get the total number of registered drivers
        
        Returns:
            Number of drivers
        """
        with self._lock:
            return len(self._drivers)

    def get_driver_names(self) -> List[str]:
        """
        Get all driver names
        
        Returns:
            List of driver names
        """
        with self._lock:
            return list(self._drivers.keys())

    def get_driver_types(self) -> List[Type]:
        """
        Get all registered driver types
        
        Returns:
            List of driver types
        """
        with self._lock:
            return list(self._drivers_by_type.keys())


    def on_initialize(self):
        """
        Called when component is initialized
        """
        logger.debug("BackendProvider initializing...")

    def on_start(self):
        """
        Called when component is started
        """
        logger.debug("BackendProvider starting...")

    def __repr__(self):
        return f"BackendProvider(drivers={len(self._drivers)})"
