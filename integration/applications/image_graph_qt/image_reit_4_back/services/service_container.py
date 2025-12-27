"""
Service Container - Single source of truth for all application services
Following Singleton pattern with lazy service instantiation.
"""
from typing import Dict, Type, Any, Optional
from loguru import logger




class ServiceContainer:
    """Central service registry implementing service locator pattern"""

    _instance: Optional['ServiceContainer'] = None

    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if ServiceContainer._instance is not None:
            raise RuntimeError("ServiceContainer is a singleton. Use get_instance() instead.")

        self._services: Dict[Type, Any] = {}
        self._service_factories: Dict[Type, callable] = {}
        self._register_services()

        ServiceContainer._instance = self
        logger.info("ServiceContainer initialized with core services")

    def register_service(self, service_type: Type, service_instance: Any):
        """Register a service instance"""
        self._services[service_type] = service_instance
        logger.debug(f"Registered service: {service_type.__name__}")

    def register_factory(self, service_type: Type, factory_func: callable):
        """Register a service factory function"""
        self._service_factories[service_type] = factory_func
        logger.debug(f"Registered factory for: {service_type.__name__}")

    def get_service(self, service_type: Type) -> Any:
        """Get service instance, creating it if necessary"""
        # Check if already instantiated
        if service_type in self._services:
            return self._services[service_type]

        # Try to create from factory
        if service_type in self._service_factories:
            try:
                service = self._service_factories[service_type]()
                self._services[service_type] = service
                logger.debug(f"Created service from factory: {service_type.__name__}")
                return service
            except Exception as e:
                logger.error(f"Failed to create service {service_type.__name__}: {e}")
                raise

        # Service not registered
        raise ValueError(f"Service {service_type.__name__} not registered")

    def has_service(self, service_type: Type) -> bool:
        """Check if service is registered or can be created"""
        return service_type in self._services or service_type in self._service_factories

    def clear_service(self, service_type: Type):
        """Clear a service instance (for testing or reset)"""
        if service_type in self._services:
            del self._services[service_type]
            logger.debug(f"Cleared service: {service_type.__name__}")

    def get_registered_services(self) -> Dict[Type, Any]:
        """Get all currently instantiated services"""
        return self._services.copy()

    def _register_services(self):
        """Register core business services"""
        # Import here to avoid circular imports
        from services.data_service import DataService
        from services.configuration_service import ConfigurationService
        from .cache_service import CacheService
        from .rating_service import RatingService
        from .validation_service import ValidationService

        # Register services with lazy instantiation
        self.register_factory(DataService, lambda: DataService())
        self.register_factory(CacheService, lambda: CacheService())
        self.register_factory(ConfigurationService, lambda: ConfigurationService())
        self.register_factory(RatingService, lambda: RatingService())
        self.register_factory(ValidationService, lambda: ValidationService())

        logger.debug("Core services registered in container")


# Global service container instance
service_container = ServiceContainer.get_instance()
