"""
Protocol definitions for Foundation plugin systems.

Provides Protocol-based interfaces for structural typing, enabling
duck-typing compatibility alongside existing ABC base classes.
"""
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from bson import ObjectId


@runtime_checkable
class ExtractorProtocol(Protocol):
    """
    Protocol for file extractors.
    
    Enables structural typing - any class implementing these methods
    is considered an ExtractorProtocol, without needing to inherit.
    
    Usage for type hints:
        def register_extractor(ext: ExtractorProtocol) -> None:
            # Works with any class that has the right methods
            pass
            
    Usage for runtime checks:
        if isinstance(obj, ExtractorProtocol):
            # True if obj has the required methods
            result = await obj.extract(files)
    
    Attributes:
        name: Unique identifier for this extractor
        phase: Processing phase (2 or 3)
        priority: Execution order (higher = first)
    """
    name: str
    phase: int
    priority: int
    
    async def extract(self, files: List[Any]) -> Dict[ObjectId, Any]:
        """Extract data from files."""
        ...
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store extracted data."""
        ...
    
    def can_process(self, file: Any) -> bool:
        """Check if this extractor can process the given file."""
        ...


@runtime_checkable
class ServiceProtocol(Protocol):
    """
    Protocol for services (systems).
    
    Enables structural typing for services without requiring BaseSystem inheritance.
    """
    is_ready: bool
    
    async def initialize(self) -> None:
        """Initialize the service."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the service."""
        ...


@runtime_checkable  
class PluginProtocol(Protocol):
    """
    Protocol for generic plugins.
    
    Base protocol for any plugin that can be discovered and loaded.
    """
    name: str
    version: str
    
    def activate(self) -> None:
        """Activate the plugin."""
        ...
    
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        ...
