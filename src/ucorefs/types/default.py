"""
UCoreFS - Default File Driver

Default driver for unknown file types.
"""
from typing import Dict, Any, Optional
from pathlib import Path

from src.ucorefs.types.driver import IFileDriver
from src.ucorefs.models.base import FSRecord


class DefaultDriver(IFileDriver):
    """
    Default driver for files with no specific handler.
    
    Provides basic metadata extraction without AI features.
    """
    
    driver_id = "default"
    supported_extensions = []  # Handles everything
    
    def can_handle(self, path: str, extension: str = None) -> bool:
        """Default driver handles everything."""
        return True
    
    async def extract_metadata(self, record: FSRecord) -> Dict[str, Any]:
        """
        Extract basic metadata.
        
        Args:
            record: FSRecord to extract from
            
        Returns:
            Basic metadata dict
        """
        metadata = {
            "file_type": "unknown",
            "mime_type": "application/octet-stream"
        }
        
        if hasattr(record, 'extension'):
            metadata["extension"] = record.extension
        
        return metadata
    
    async def get_thumbnail(self, record: FSRecord) -> Optional[bytes]:
        """Default driver doesn't generate thumbnails."""
        return None
