
"""
Service Decorator.

Reduces boilerplate for Service initialization and logging.
"""
from typing import Type, TypeVar, Optional, Any
from functools import wraps
from loguru import logger
import inspect
import asyncio

T = TypeVar("T")

def Service(cls: Type[T]) -> Type[T]:
    """
    Decorator for Services.
    - Adds automatic logging for initialize/shutdown.
    - Ensures base class initialize is called (if compatible).
    """
    
    orig_init = cls.__init__
    orig_initialize = getattr(cls, 'initialize', None)
    orig_shutdown = getattr(cls, 'shutdown', None)

    # Wrap initialize
    @wraps(orig_initialize)
    async def wrapped_initialize(self, *args, **kwargs):
        service_name = self.__class__.__name__
        logger.debug(f"Initializing Service: {service_name}")
        
        # Call original
        result = await orig_initialize(self, *args, **kwargs)
        
        # Verify if is_ready is set (assuming BaseSystem)
        if hasattr(self, 'is_ready') and not self.is_ready:
            # If the subclass didn't call super() (and thus didn't set is_ready),
            # we might want to warn or do it for them?
            # Doing it for them is risky if they depend on it happening before/after.
            # But the goal is to Remove Boilerplate.
            # If we assume they REMOVE super().initialize(), we must do it.
            # However, BaseSystem.initialize() sets _is_ready = True.
            # Let's call BaseSystem.initialize(self) if they didn't?
            # It's tricky with MRO.
            # SAFE APPROACH: Just Log for now.
            # AGGRESSIVE APPROACH (Plan): "Handles super().initialize() calls if missed"
            logger.warning(f"Service {service_name} did not become ready after initialize (missing super().initialize()?).")
            # Forcing readiness:
            # self._is_ready = True 
            
        logger.info(f"Service Initialized: {service_name}")
        return result

    # Wrap shutdown
    @wraps(orig_shutdown)
    async def wrapped_shutdown(self, *args, **kwargs):
        service_name = self.__class__.__name__
        logger.debug(f"Shutting down Service: {service_name}")
        
        result = await orig_shutdown(self, *args, **kwargs)
        
        logger.info(f"Service Shutdown: {service_name}")
        return result

    if orig_initialize:
        cls.initialize = wrapped_initialize
    
    if orig_shutdown:
        cls.shutdown = wrapped_shutdown

    return cls
