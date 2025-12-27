"""
Driver Registry for AI Engines.

Modular backend registry supporting multiple implementations (realizations)
for each AI capability (embedding, detection, etc.).
"""
from typing import Dict, List, Optional, Type, TypeVar, Callable
from abc import ABC, abstractmethod
from loguru import logger

from src.core.base_system import BaseSystem

T = TypeVar('T', bound='AIDriver')


class AIDriver(ABC):
    """
    Base class for AI drivers.
    
    A driver represents a capability (e.g., "embedding", "detection")
    that can have multiple implementations (realizations).
    
    Usage:
        class EmbeddingDriver(AIDriver):
            def on_load(self):
                self.register_realization("clip", ClipEmbedder())
                self.register_realization("blip", BlipEmbedder())
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the driver.
        
        Args:
            name: Driver name
        """
        self.name = name or self.__class__.__name__
        self.version = "1.0.0"
        self._realizations: Dict[str, object] = {}
        self._active_realization: Optional[str] = None
        self._is_loaded = False
    
    def register_realization(self, name: str, instance: object, tags: List[str] = None) -> None:
        """
        Register a realization (implementation).
        
        Args:
            name: Realization name (e.g., "gpu", "cpu", "clip")
            instance: Implementation instance
            tags: Optional tags for filtering
        """
        self._realizations[name] = {
            "instance": instance,
            "tags": tags or []
        }
        logger.debug(f"Registered realization: {self.name}.{name}")
    
    def get_realization(self, name: str) -> Optional[object]:
        """
        Get a realization by name.
        
        Args:
            name: Realization name
            
        Returns:
            Implementation instance or None
        """
        entry = self._realizations.get(name)
        return entry["instance"] if entry else None
    
    def get_realization_by_tag(self, tag: str) -> List[object]:
        """
        Get realizations by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching implementations
        """
        return [
            entry["instance"]
            for entry in self._realizations.values()
            if tag in entry.get("tags", [])
        ]
    
    def get_active(self) -> Optional[object]:
        """Get the active realization."""
        if self._active_realization:
            return self.get_realization(self._active_realization)
        return None
    
    def set_active(self, name: str) -> bool:
        """
        Set the active realization.
        
        Args:
            name: Realization name
            
        Returns:
            True if set successfully
        """
        if name in self._realizations:
            self._active_realization = name
            logger.info(f"{self.name}: Active realization set to '{name}'")
            return True
        return False
    
    def list_realizations(self) -> List[str]:
        """List all realization names."""
        return list(self._realizations.keys())
    
    def load(self, force: bool = False) -> bool:
        """Load the driver."""
        if self._is_loaded and not force:
            return True
        
        try:
            self.on_load()
            self._is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load driver {self.name}: {e}")
            return False
    
    @abstractmethod
    def on_load(self) -> None:
        """Called when driver is loaded. Register realizations here."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class DriverRegistry(BaseSystem):
    """
    Central registry for AI drivers.
    
    Manages driver registration, discovery, and access.
    
    Usage:
        registry = locator.get_system(DriverRegistry)
        registry.register(EmbeddingDriver())
        embedding = registry.get("EmbeddingDriver").get_active()
    """
    
    async def initialize(self):
        """Initialize the driver registry."""
        self._drivers: Dict[str, AIDriver] = {}
        self._is_ready = True
        logger.info("DriverRegistry initialized")
    
    async def shutdown(self):
        """Shutdown the driver registry."""
        self._drivers.clear()
        self._is_ready = False
    
    def register(self, driver: AIDriver) -> None:
        """
        Register a driver.
        
        Args:
            driver: AIDriver instance
        """
        driver.load()
        self._drivers[driver.name] = driver
        logger.info(f"Registered driver: {driver.name}")
    
    def get(self, name: str) -> Optional[AIDriver]:
        """
        Get a driver by name.
        
        Args:
            name: Driver name
            
        Returns:
            AIDriver instance or None
        """
        return self._drivers.get(name)
    
    def list_drivers(self) -> List[str]:
        """List all registered driver names."""
        return list(self._drivers.keys())
    
    def get_all(self) -> List[AIDriver]:
        """Get all registered drivers."""
        return list(self._drivers.values())
