
from typing import Protocol, List, Dict, Any, runtime_checkable
from bson import ObjectId
from src.ucorefs.models.file_record import FileRecord

@runtime_checkable
class ExtractorProtocol(Protocol):
    """
    Protocol definition for UCoreFS Extractors.
    Allows for structural subtyping instead of strict inheritance.
    """
    
    name: str
    phase: int
    priority: int
    batch_supported: bool
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Extract metadata from files."""
        ...
        
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store extraction result."""
        ...
        
    def can_process(self, file: FileRecord) -> bool:
        """Check if file is supported."""
        ...


@runtime_checkable
class IExtractorRegistry(Protocol):
    """
    Protocol for ExtractorRegistry to break circular dependencies.
    
    This protocol defines the interface that ProcessingPipeline depends on,
    allowing the pipeline to import only the protocol (not the concrete class)
    at module level, eliminating circular import issues.
    
    The concrete ExtractorRegistry class implements this protocol through
    structural subtyping (duck typing), so no explicit inheritance is needed.
    """
    
    @classmethod
    def get_for_phase(
        cls, 
        phase: int, 
        locator=None,
        config: Dict[str, dict] = None
    ) -> List[Any]:
        """
        Get extractor instances for a processing phase.
        
        Args:
            phase: Processing phase (2 or 3)
            locator: ServiceLocator for dependency injection
            config: Config dict keyed by extractor name
            
        Returns:
            List of extractor instances, sorted by priority (descending)
        """
        ...
    
    @classmethod
    def list_registered(cls) -> Dict[int, List[str]]:
        """
        List all registered extractor names by phase.
        
        Returns:
            Dict mapping phase number to list of extractor names
        """
        ...

