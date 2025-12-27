"""
UCoreFS - File Type Registry

Registry for file type drivers with factory pattern.
"""
from typing import Dict, Type, Optional, List
from pathlib import Path
from loguru import logger

from src.ucorefs.types.driver import IFileDriver


class FileTypeRegistry:
    """
    Registry for file type drivers.
    
    Provides factory pattern for getting appropriate driver
    based on file extension or path.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._drivers: Dict[str, Type[IFileDriver]] = {}
        self._instances: Dict[str, IFileDriver] = {}
    
    def register(
        self,
        driver_class: Type[IFileDriver],
        extensions: Optional[List[str]] = None
    ) -> None:
        """
        Register a driver for file extensions.
        
        Args:
            driver_class: Driver class to register
            extensions: List of extensions (uses driver.supported_extensions if None)
        """
        exts = extensions or driver_class.supported_extensions
        
        for ext in exts:
            ext_lower = ext.lower().lstrip('.')
            self._drivers[ext_lower] = driver_class
            logger.debug(f"Registered {driver_class.driver_id} for .{ext_lower}")
    
    def get_driver(self, path: str = None, extension: str = None) -> IFileDriver:
        """
        Get driver instance for a file.
        
        Args:
            path: File path (optional)
            extension: File extension (optional, extracted from path if not provided)
            
        Returns:
            IFileDriver instance (default driver if none registered)
        """
        # Determine extension
        ext = extension
        if not ext and path:
            ext = Path(path).suffix.lstrip('.')
        
        if not ext:
            return self._get_default_driver()
        
        ext_lower = ext.lower()
        
        # Get driver class
        driver_class = self._drivers.get(ext_lower)
        if not driver_class:
            return self._get_default_driver()
        
        # Get or create instance
        driver_id = driver_class.driver_id
        if driver_id not in self._instances:
            self._instances[driver_id] = driver_class()
        
        return self._instances[driver_id]
    
    def get_driver_by_id(self, driver_id: str) -> Optional[IFileDriver]:
        """
        Get driver instance by ID.
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            IFileDriver instance or None
        """
        return self._instances.get(driver_id)
    
    def _get_default_driver(self) -> IFileDriver:
        """Get default driver for unknown file types."""
        from src.ucorefs.types.default import DefaultDriver
        
        if "default" not in self._instances:
            self._instances["default"] = DefaultDriver()
        
        return self._instances["default"]
    
    def list_supported_extensions(self) -> List[str]:
        """Get list of all supported extensions."""
        return list(self._drivers.keys())
    
    def clear(self) -> None:
        """Clear all registered drivers."""
        self._drivers.clear()
        self._instances.clear()


# Global registry instance
registry = FileTypeRegistry()
