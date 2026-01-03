"""
Context managers for Foundation services.

Provides async context managers for cleaner service lifecycle handling in tests and scripts.
"""
from contextlib import asynccontextmanager
from typing import Type, TypeVar
from loguru import logger

from .base_system import BaseSystem
from .locator import sl

T = TypeVar('T', bound=BaseSystem)


@asynccontextmanager
async def managed_service(service_cls: Type[T]) -> T:
    """
    Async context manager for service lifecycle.
    
    Automatically initializes service on enter and shuts down on exit.
    Useful for tests and standalone scripts.
    
    Example:
        async with managed_service(FSService) as fs:
            files = await fs.get_files()
            
    Args:
        service_cls: Service class to manage
        
    Yields:
        Initialized service instance
    """
    # Get or create service instance
    try:
        service = sl.get_system(service_cls)
    except KeyError:
        # Not registered, create new instance
        service = sl.register_system(service_cls)
    
    # Initialize if not ready
    if not service.is_ready:
        await service.initialize()
        logger.debug(f"managed_service: Initialized {service_cls.__name__}")
    
    try:
        yield service
    finally:
        # Shutdown on exit
        if service.is_ready:
            await service.shutdown()
            logger.debug(f"managed_service: Shutdown {service_cls.__name__}")


@asynccontextmanager
async def managed_locator(config_path: str = "config.json"):
    """
    Async context manager for full ServiceLocator lifecycle.
    
    Initializes the locator and starts all registered systems,
    then shuts them all down on exit.
    
    Example:
        async with managed_locator("config.json"):
            fs = sl.get_system(FSService)
            await fs.do_work()
            
    Args:
        config_path: Path to config file
        
    Yields:
        Initialized ServiceLocator
    """
    sl.init(config_path)
    await sl.start_all()
    logger.info("managed_locator: All systems started")
    
    try:
        yield sl
    finally:
        await sl.stop_all()
        logger.info("managed_locator: All systems stopped")
