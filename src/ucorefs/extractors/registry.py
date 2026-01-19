"""
UCoreFS - Extractor Registry

Central registry for file extractors.
Implements plugin architecture for extensibility (SOLID: OCP).
"""
from typing import Dict, List, Type, Optional
from loguru import logger

from src.ucorefs.extractors.base import Extractor


class ExtractorRegistry:
    """
    Central registry for all file extractors.
    
    Extractors register themselves at application startup.
    ProcessingPipeline queries this registry to get extractors per phase.
    
    Note: This class implements IExtractorRegistry protocol via structural
    subtyping (duck typing). No explicit inheritance is required; the class
    satisfies the protocol by having the required methods with matching signatures.
    
    Example:
        # Register extractors
        ExtractorRegistry.register(ThumbnailExtractor)
        ExtractorRegistry.register(MetadataExtractor)
        ExtractorRegistry.register(CLIPExtractor)
        
        # Get extractors for Phase 2
        phase2_extractors = ExtractorRegistry.get_for_phase(2, locator)
    """
    
    _extractors: Dict[int, List[Type[Extractor]]] = {
        2: [],  # Phase 2 extractors
        3: [],  # Phase 3 extractors
    }
    
    _instances: Dict[str, Extractor] = {}  # Cached instances
    
    @classmethod
    def register(cls, extractor_cls: Type[Extractor]) -> None:
        """
        Register an extractor class.
        
        Args:
            extractor_cls: Extractor class to register
        """
        phase = getattr(extractor_cls, 'phase', 2)
        name = getattr(extractor_cls, 'name', extractor_cls.__name__)
        
        if phase not in cls._extractors:
            cls._extractors[phase] = []
        
        # Avoid duplicates
        if extractor_cls not in cls._extractors[phase]:
            cls._extractors[phase].append(extractor_cls)
            logger.debug(f"Registered extractor: {name} (phase {phase})")
    
    @classmethod
    def unregister(cls, extractor_cls: Type[Extractor]) -> None:
        """Remove extractor from registry."""
        phase = getattr(extractor_cls, 'phase', 2)
        if phase in cls._extractors and extractor_cls in cls._extractors[phase]:
            cls._extractors[phase].remove(extractor_cls)
    
    @classmethod
    def get_for_phase(
        cls, 
        phase: int, 
        locator=None,
        config: Dict[str, dict] = None
    ) -> List[Extractor]:
        """
        Get extractor instances for a processing phase.
        
        Args:
            phase: Processing phase (2 or 3)
            locator: ServiceLocator for dependency injection
            config: Config dict keyed by extractor name
            
        Returns:
            List of extractor instances, sorted by priority (descending)
        """
        config = config or {}
        instances = []
        
        for extractor_cls in cls._extractors.get(phase, []):
            name = getattr(extractor_cls, 'name', extractor_cls.__name__)
            
            instance = None
            
            # 1. Try resolving via ServiceLocator (preferred for Singletons/persistence)
            if locator and hasattr(locator, 'get_system'):
                try:
                    instance = locator.get_system(extractor_cls)
                    # Cache it for get_by_name compatibility
                    cls._instances[name] = instance
                except (KeyError, AttributeError):
                    pass
            
            # 2. Use locally cached instance
            if instance is None and name in cls._instances:
                instance = cls._instances[name]
                
            # 3. Create new instance (fallback)
            if instance is None:
                extractor_config = config.get(name, {})
                instance = extractor_cls(locator=locator, config=extractor_config)
                cls._instances[name] = instance
                
            instances.append(instance)
        
        # Sort by priority (higher first)
        instances.sort(key=lambda e: e.priority, reverse=True)
        
        return instances
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[Extractor]:
        """Get an extractor instance by name."""
        return cls._instances.get(name)
    
    @classmethod
    def list_registered(cls) -> Dict[int, List[str]]:
        """List all registered extractor names by phase."""
        return {
            phase: [getattr(e, 'name', e.__name__) for e in extractors]
            for phase, extractors in cls._extractors.items()
        }
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._extractors = {2: [], 3: []}
        cls._instances = {}
