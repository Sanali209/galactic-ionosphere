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
    needs_model: bool = False   # If True, requires AI model (not suitable for subprocess)
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        """
        Initialize extractor.
        
        Args:
            locator: ServiceLocator for accessing other services
            config: Configuration dict for this extractor
        """
        self.locator = locator
        self.config = config or {}
    
    async def initialize(self) -> None:
        """
        Initialize the extractor.
        
        Called by ServiceLocator when registered as a system.
        Override to perform async setup (e.g. model loading).
        """
        pass
        
    async def shutdown(self) -> None:
        """
        Shutdown the extractor.
        
        Called by ServiceLocator on app exit.
        Override to cleanup resources (e.g. unload models).
        """
        pass
    
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
        import asyncio
        
        results = {}
        
        try:
            extracted = await self.extract(files)
            
            if not extracted:
                logger.warning(f"{self.name}: extract() returned empty results for {len(files)} files")
            
            for i, (file_id, data) in enumerate(extracted.items()):
                try:
                    success = await self.store(file_id, data)
                    results[file_id] = success
                    if not success:
                        logger.error(f"{self.name}: Storage returned False for {file_id}")
                except Exception as e:
                    logger.error(f"{self.name}: Failed to store for {file_id}: {e}", exc_info=True)
                    results[file_id] = False
                
                # Yield to UI thread every 5 stores to prevent blocking
                if (i + 1) % 5 == 0:
                    await asyncio.sleep(0)
                    
        except RuntimeError as e:
            # Catch executor shutdown errors during application exit
            if "shutdown" in str(e).lower():
                logger.debug(f"{self.name}: Skipping extraction - executor shutdown in progress")
                # Return empty results - task will stay 'running' and be recovered on restart
                return {}
            else:
                logger.error(f"{self.name}: Extraction failed: {e}", exc_info=True)
                for file in files:
                    results[file._id] = False
        except Exception as e:
            logger.error(f"{self.name}: Extraction failed: {e}", exc_info=True)
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
