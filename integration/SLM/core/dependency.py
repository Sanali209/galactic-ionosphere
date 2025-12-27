"""
Dependency Injection System
Manages service dependencies and provides dependency injection
"""

from typing import Dict, Any, Optional, Type, TypeVar, Callable, List, Set
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from loguru import logger

from SLM.core.singleton import Singleton

T = TypeVar('T')


class ServiceProvider(ABC):
    """
    Service provider interface
    """

    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service instance

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance
        """
        pass

    @abstractmethod
    def register_service(self, service_type: Type[T], service_instance: T):
        """
        Register a service instance

        Args:
            service_type: Service type
            service_instance: Service instance
        """
        pass


class DependencyContainer(ServiceProvider):
    """
    Container for managing service dependencies
    """

    def __init__(self):
        """
        Initialize the dependency container
        """
        # Prevent re-initialization if already initialized
        if hasattr(self, '_initialized'):
            return
            
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.Lock()
        self._initialized = True

    def register_service(self, service_type: Type[T], service_instance: T):
        """
        Register a service instance

        Args:
            service_type: Service type
            service_instance: Service instance
        """
        with self._lock:
            self._services[service_type] = service_instance
            logger.info(f"Registered service: {service_type.__name__}")

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]):
        """
        Register a service factory

        Args:
            service_type: Service type
            factory: Factory function that creates service instances
        """
        with self._lock:
            self._factories[service_type] = factory
            logger.info(f"Registered factory for: {service_type.__name__}")

    def register_singleton(self, service_type: Type[T], service_instance: T):
        """
        Register a singleton service

        Args:
            service_type: Service type
            service_instance: Service instance
        """
        with self._lock:
            self._singletons[service_type] = service_instance
            logger.info(f"Registered singleton: {service_type.__name__}")

    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service instance

        Args:
            service_type: Service type

        Returns:
            Service instance
        """
        with self._lock:
            # Check singletons first
            if service_type in self._singletons:
                return self._singletons[service_type]

            # Check registered services
            if service_type in self._services:
                return self._services[service_type]

            # Check factories
            if service_type in self._factories:
                factory = self._factories[service_type]
                instance = factory()
                # Cache the instance for future use
                self._services[service_type] = instance
                return instance

            raise KeyError(f"No service registered for type {service_type.__name__}")

    def has_service(self, service_type: Type[T]) -> bool:
        """
        Check if a service is registered

        Args:
            service_type: Service type

        Returns:
            True if service is registered
        """
        with self._lock:
            return (service_type in self._services or
                   service_type in self._factories or
                   service_type in self._singletons)

    def remove_service(self, service_type: Type[T]):
        """
        Remove a service registration

        Args:
            service_type: Service type
        """
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
            if service_type in self._factories:
                del self._factories[service_type]
            if service_type in self._singletons:
                del self._singletons[service_type]

    def clear(self):
        """
        Clear all service registrations
        """
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()

    def get_service_count(self) -> int:
        """
        Get the total number of registered services

        Returns:
            Number of services
        """
        with self._lock:
            return (len(self._services) +
                   len(self._factories) +
                   len(self._singletons))

    def get_service_types(self) -> list:
        """
        Get all registered service types, filtering out abstract classes.

        Returns:
            List of concrete service types.
        """
        with self._lock:
            all_types = set()
            all_types.update(self._services.keys())
            all_types.update(self._factories.keys())
            all_types.update(self._singletons.keys())
            
            # Filter out abstract classes
            concrete_types = [
                t for t in all_types 
                if not (isinstance(t, type) and getattr(t, '__isabstractmethod__', False))
            ]
            return concrete_types


class DependencyInjector:
    """
    Handles dependency injection for classes
    """

    def __init__(self, container: DependencyContainer):
        """
        Initialize the dependency injector

        Args:
            container: Service container
        """
        self.container = container

    def inject_dependencies(self, obj: Any):
        """
        Inject dependencies into an object

        Args:
            obj: Object to inject dependencies into
        """
        # Look for type annotations that match registered services
        if hasattr(obj, '__annotations__'):
            for attr_name, service_type in obj.__annotations__.items():
                if self.container.has_service(service_type):
                    try:
                        service_instance = self.container.get_service(service_type)
                        setattr(obj, attr_name, service_instance)
                    except Exception as e:
                        logger.error(f"Failed to inject dependency {service_type.__name__} into {obj.__class__.__name__}: {e}")

    def create_instance(self, cls: Type[T], *args, **kwargs) -> T:
        """
        Create an instance of a class with dependency injection

        Args:
            cls: Class to instantiate
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Instance with dependencies injected
        """
        instance = cls(*args, **kwargs)
        self.inject_dependencies(instance)
        return instance


class DependencyGraph:
    """
    Analyzes service dependencies from type annotations
    """

    def __init__(self, service_types: List[Type], container: DependencyContainer):
        """
        Initialize dependency graph

        Args:
            service_types: List of service types to analyze
            container: Service container for checking available services
        """
        self.service_types = service_types
        self.container = container
        self.graph: Dict[Type, Set[Type]] = defaultdict(set)  # service -> dependencies
        self.reverse_graph: Dict[Type, Set[Type]] = defaultdict(set)  # service -> dependents
        self._build_graph()

    def _build_graph(self):
        """
        Build dependency graph from type annotations
        """
        for service_type in self.service_types:
            if hasattr(service_type, '__annotations__'):
                for attr_name, dep_type in service_type.__annotations__.items():
                    if self.container.has_service(dep_type) and dep_type in self.service_types:
                        self.graph[service_type].add(dep_type)
                        self.reverse_graph[dep_type].add(service_type)

    def get_dependencies(self, service_type: Type) -> Set[Type]:
        """
        Get direct dependencies of a service

        Args:
            service_type: Service type

        Returns:
            Set of dependency types
        """
        return self.graph.get(service_type, set())

    def has_cycles(self) -> bool:
        """
        Check if the dependency graph has cycles

        Returns:
            True if cycles detected
        """
        visited = set()
        rec_stack = set()

        def dfs(node: Type) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for dep in self.graph[node]:
                if dfs(dep):
                    return True

            rec_stack.remove(node)
            return False

        for service_type in self.service_types:
            if service_type not in visited:
                if dfs(service_type):
                    return True
        return False

    def get_initialization_order(self) -> List[Type]:
        """
        Get services in topological initialization order

        Returns:
            List of service types in initialization order

        Raises:
            ValueError: If circular dependencies detected
        """
        if self.has_cycles():
            # Find cycles for better error reporting
            cycles = self._find_cycles()
            raise ValueError(f"Circular dependencies detected: {cycles}")

        # Kahn's algorithm for topological sort
        # We need to initialize services that have no dependencies first
        order = []
        in_degree = {node: len(self.graph[node]) for node in self.service_types}  # Number of dependencies
        queue = deque()

        # Start with services that have no dependencies (in_degree = 0)
        for node in self.service_types:
            if in_degree[node] == 0:
                queue.append(node)

        while queue:
            current = queue.popleft()
            order.append(current)

            # For each service that depends on current, reduce their dependency count
            for dependent in self.reverse_graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for remaining nodes (should not happen if no cycles)
        if len(order) != len(self.service_types):
            remaining = set(self.service_types) - set(order)
            raise ValueError(f"Failed to resolve dependencies for: {[t.__name__ for t in remaining]}")

        return order

    def _find_cycles(self) -> List[List[Type]]:
        """
        Find all cycles in the dependency graph

        Returns:
            List of cycles (each cycle is a list of types)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: Type, path: List[Type]) -> bool:
            if node in rec_stack:
                # Found cycle - extract from current path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            has_cycle = False
            for dep in self.graph[node]:
                if dfs(dep, path):
                    has_cycle = True
                    # Continue looking for more cycles

            path.pop()
            rec_stack.remove(node)
            return has_cycle

        for service_type in self.service_types:
            if service_type not in visited:
                dfs(service_type, [])

        return cycles


class DependencyManager(Singleton):
    """
    Singleton dependency management system
    """

    def __init__(self):
        """
        Initialize the dependency manager
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
            
        self.container = DependencyContainer()
        self.injector = DependencyInjector(self.container)
        self._initialized = True

    def register_service(self, service_type: Type[T], service_instance: T):
        """
        Register a service instance

        Args:
            service_type: Service type
            service_instance: Service instance
        """
        self.container.register_service(service_type, service_instance)

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]):
        """
        Register a service factory

        Args:
            service_type: Service type
            factory: Factory function
        """
        self.container.register_factory(service_type, factory)

    def register_singleton(self, service_type: Type[T], service_instance: T):
        """
        Register a singleton service

        Args:
            service_type: Service type
            service_instance: Service instance
        """
        self.container.register_singleton(service_type, service_instance)

    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service instance

        Args:
            service_type: Service type

        Returns:
            Service instance
        """
        return self.container.get_service(service_type)

    def has_service(self, service_type: Type[T]) -> bool:
        """
        Check if a service is registered

        Args:
            service_type: Service type

        Returns:
            True if service is registered
        """
        return self.container.has_service(service_type)

    def setup_dependencies(self):
        """
        Setup automatic dependency injection and initialization in correct dependency order
        """
        logger.info("Setting up dependencies...")

        # Get all service types that need dependency analysis
        all_service_types = self.get_service_types()
        service_types_to_process = [
            st for st in all_service_types
            if st.__name__ not in ['App']  # Skip App to avoid circular dependency
        ]

        # Build dependency graph for ordering
        dependency_graph = DependencyGraph(service_types_to_process, self.container)
        initialization_order = dependency_graph.get_initialization_order()

        logger.info(f"Dependency analysis complete. Initialization order: {[t.__name__ for t in initialization_order]}")

        # Phase 1: Inject dependencies into ALL services first
        logger.debug("Phase 1: Injecting dependencies...")
        for service_type in service_types_to_process:
            try:
                service = self.get_service(service_type)
                if hasattr(service_type, '__annotations__'):  # Check class annotations
                    for attr_name, dep_type in service_type.__annotations__.items():
                        if self.has_service(dep_type) and dep_type in service_types_to_process:
                            # Inject if attribute doesn't exist OR is None (allows overriding None defaults)
                            if not hasattr(service, attr_name) or getattr(service, attr_name, None) is None:
                                dep_instance = self.get_service(dep_type)
                                setattr(service, attr_name, dep_instance)
                                logger.debug(f"Injected {dep_type.__name__} into {service_type.__name__}.{attr_name}")

            except Exception as e:
                logger.error(f"Error injecting dependencies for {service_type.__name__}: {e}")

        # Phase 2: Initialize services in dependency order
        logger.debug("Phase 2: Initializing services in dependency order...")
        for service_type in initialization_order:
            try:
                service = self.get_service(service_type)

                # Initialize service if not already initialized
                if hasattr(service, 'initialize') and callable(service.initialize):
                    if not getattr(service, '_initialized', False):
                        service.initialize()
                        service._initialized = True
                        logger.debug(f"Initialized {service_type.__name__}")

            except Exception as e:
                logger.error(f"Error initializing {service_type.__name__}: {e}")
                raise

    def inject_dependencies(self, obj: Any):
        """
        Inject dependencies into an object

        Args:
            obj: Object to inject dependencies into
        """
        self.injector.inject_dependencies(obj)

    def create_instance(self, cls: Type[T], *args, **kwargs) -> T:
        """
        Create an instance with dependency injection

        Args:
            cls: Class to instantiate
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Instance with dependencies injected
        """
        return self.injector.create_instance(cls, *args, **kwargs)

    def clear(self):
        """
        Clear all service registrations
        """
        self.container.clear()

    def get_service_count(self) -> int:
        """
        Get the total number of registered services

        Returns:
            Number of services
        """
        return self.container.get_service_count()

    def get_service_types(self) -> list:
        """
        Get all registered service types

        Returns:
            List of service types
        """
        return self.container.get_service_types()

    def __repr__(self):
        return f"DependencyManager(services={self.get_service_count()})"

    def __str__(self):
        return f"DependencyManager managing {self.get_service_count()} services"

    def initialize_all(self):
        """
        Initialize all services that have an 'initialize' method
        """
        for service_type in self.get_service_types():
            service = self.get_service(service_type)
            if hasattr(service, 'initialize') and callable(service.initialize):
                try:
                    if not getattr(service, '_initialized', False):
                        service.initialize()
                        service._initialized = True
                        logger.debug(f"Initialized service: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to initialize service {service_type.__name__}: {e}")

    def start_all(self):
        """
        Start all services that have a 'start' method
        Excludes the App service which should be started manually
        """
        for service_type in self.get_service_types():
            # Skip the App class - it should be started externally, not by dependency manager
            if service_type.__name__ == 'App':
                continue

            service = self.get_service(service_type)
            if hasattr(service, 'start') and callable(service.start):
                try:
                    service.start()
                    logger.debug(f"Started service: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to start service {service_type.__name__}: {e}")

    def stop_all(self):
        """
        Stop all services that have a 'stop' method
        """
        for service_type in self.get_service_types():
            service = self.get_service(service_type)
            if hasattr(service, 'stop') and callable(service.stop):
                try:
                    service.stop()
                    logger.debug(f"Stopped service: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to stop service {service_type.__name__}: {e}")

    def update_all(self):
        """
        Update all services that have an 'update' method
        """
        for service_type in self.get_service_types():
            # Skip Config and App classes - they don't need periodic updates
            if service_type.__name__ in ['Config', 'App']:
                continue
                
            service = self.get_service(service_type)
            if hasattr(service, 'update') and callable(service.update):
                try:
                    service.update()
                except Exception as e:
                    logger.error(f"Failed to update service {service_type.__name__}: {e}")
