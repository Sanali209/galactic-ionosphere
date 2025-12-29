"""
UCoreFS - Extractor Base Class

Abstract base class for all file extractors.
Extractors process files in Phase 2/3 of the processing pipeline.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from bson import ObjectId
from loguru import logger

from src.ucorefs.models.file_record import FileRecord


class Extractor(ABC):
    """
    Base class for all file extractors.
    
    Extractors are plugins that process files during Phase 2/3.
    Each extractor has a single responsibility (SOLID: SRP).
    
    Attributes:
        name: Unique identifier for this extractor
        phase: Processing phase (2 or 3)
        priority: Execution order (higher = first)
        batch_supported: Whether batch processing is supported
        is_cpu_heavy: Whether this extractor does CPU-intensive work (SAN-14)
                     If True, implementations should use thread pool offloading
    """
    
    name: str = "base_extractor"
    phase: int = 2
    priority: int = 0
    batch_supported: bool = True
    is_cpu_heavy: bool = False  # SAN-14: Flag for CPU-intensive extractors
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        """
        Initialize extractor.
        
        Args:
            locator: ServiceLocator for accessing other services
            config: Configuration dict for this extractor
        """
        self.locator = locator
        self.config = config or {}
    
    @abstractmethod
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Extract data from files.
        
        Args:
            files: List of FileRecord objects to process
            
        Returns:
            Dict mapping file_id -> extracted result
        """
        pass
    
    @abstractmethod
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """
        Store extracted data.
        
        Args:
            file_id: File ObjectId
            result: Extraction result to store
            
        Returns:
            True if storage succeeded
        """
        pass
    
    async def process(self, files: List[FileRecord]) -> Dict[ObjectId, bool]:
        """
        Full processing: extract + store.
        
        Args:
            files: Files to process
            
        Returns:
            Dict mapping file_id -> success status
        """
        results = {}
        
        try:
            extracted = await self.extract(files)
            
            for file_id, data in extracted.items():
                try:
                    success = await self.store(file_id, data)
                    results[file_id] = success
                except Exception as e:
                    logger.error(f"{self.name}: Failed to store for {file_id}: {e}")
                    results[file_id] = False
                    
        except Exception as e:
            logger.error(f"{self.name}: Extraction failed: {e}")
            for file in files:
                results[file._id] = False
        
        return results
    
    def can_process(self, file: FileRecord) -> bool:
        """
        Check if this extractor can process the given file.
        
        Override to filter by file type, extension, etc.
        
        Args:
            file: File to check
            
        Returns:
            True if extractor can process this file
        """
        return True
