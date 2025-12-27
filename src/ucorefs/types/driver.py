"""
UCoreFS - File Driver Interface

Abstract base class for file type handlers with AI capabilities.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from bson import ObjectId

from src.ucorefs.models.base import FSRecord


class IFileDriver(ABC):
    """
    Interface for file type drivers.
    
    Drivers handle file-specific operations including:
    - Metadata extraction
    - Thumbnail generation
    - AI processing (embeddings, captions, descriptions)
    - Virtual file navigation (for packages/archives)
    """
    
    driver_id: str = "base"
    supported_extensions: List[str] = []
    
    @abstractmethod
    def can_handle(self, path: str, extension: str = None) -> bool:
        """
        Check if this driver can handle the given file.
        
        Args:
            path: File path
            extension: File extension (optional)
            
        Returns:
            True if driver can handle this file
        """
        pass
    
    @abstractmethod
    async def extract_metadata(self, record: FSRecord) -> Dict[str, Any]:
        """
        Extract file-specific metadata.
        
        Args:
            record: FSRecord to extract metadata from
            
        Returns:
            Dictionary of metadata fields
        """
        pass
    
    @abstractmethod
    async def get_thumbnail(self, record: FSRecord) -> Optional[bytes]:
        """
        Generate thumbnail for the file.
        
        Args:
            record: FSRecord to generate thumbnail for
            
        Returns:
            Thumbnail bytes or None
        """
        pass
    
    async def get_children(self, record: FSRecord) -> List[FSRecord]:
        """
        Get virtual children (for packages/archives).
        
        Args:
            record: Parent FSRecord
            
        Returns:
            List of child FSRecords (empty by default)
        """
        return []
    
    # ==================== AI Processing Methods ====================
    
    async def get_embedding_vector(self, record: FSRecord) -> Optional[List[float]]:
        """
        Get primary embedding vector for the file.
        
        Args:
            record: FSRecord to generate embedding for
            
        Returns:
            Embedding vector or None
        """
        return None
    
    async def get_clip_embedding(self, record: FSRecord) -> Optional[List[float]]:
        """
        Generate CLIP embedding (for images/video).
        
        Args:
            record: FSRecord to process
            
        Returns:
            CLIP vector or None
        """
        return None
    
    async def get_blip_caption(self, record: FSRecord) -> Optional[str]:
        """
        Generate BLIP caption (for images).
        
        Args:
            record: FSRecord to caption
            
        Returns:
            Caption text or None
        """
        return None
    
    async def find_similar(
        self,
        record: FSRecord,
        threshold: float = 0.85
    ) -> List[ObjectId]:
        """
        Find similar files using embeddings.
        
        Args:
            record: FSRecord to find similar files for
            threshold: Similarity threshold
            
        Returns:
            List of similar file ObjectIds
        """
        return []
    
    async def generate_llm_description(self, record: FSRecord) -> Optional[str]:
        """
        Generate LLM description from file/thumbnail.
        
        Args:
            record: FSRecord to describe
            
        Returns:
            Generated description or None
        """
        return None
