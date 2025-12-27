"""
UCoreFS - Text File Driver

Driver for text-based files.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

from src.ucorefs.types.driver import IFileDriver
from src.ucorefs.models.base import FSRecord


class TextDriver(IFileDriver):
    """
    Driver for text files.
    
    Features:
    - Encoding detection
    - Line count
    - Text embeddings (for search)
    """
    
    driver_id = "text"
    supported_extensions = ["txt", "md", "json", "xml", "csv", "log"]
    
    def can_handle(self, path: str, extension: str = None) -> bool:
        """Check if this is a text file."""
        ext = extension or Path(path).suffix.lstrip('.').lower()
        return ext in self.supported_extensions
    
    async def extract_metadata(self, record: FSRecord) -> Dict[str, Any]:
        """
        Extract text file metadata.
        
        Args:
            record: FileRecord to extract from
            
        Returns:
            Metadata dict
        """
        metadata = {
            "file_type": "text",
            "mime_type": "text/plain",
            "encoding": "utf-8",
            "line_count": 0
        }
        
        try:
            # Detect encoding and count lines
            with open(record.path, 'rb') as f:
                content = f.read()
            
            # Try UTF-8 first
            try:
                text = content.decode('utf-8')
                metadata["encoding"] = "utf-8"
            except UnicodeDecodeError:
                # Fallback to latin-1
                text = content.decode('latin-1', errors='ignore')
                metadata["encoding"] = "latin-1"
            
            metadata["line_count"] = text.count('\n') + 1
        
        except Exception as e:
            logger.error(f"Failed to extract text metadata from {record.path}: {e}")
        
        return metadata
    
    async def get_thumbnail(self, record: FSRecord) -> Optional[bytes]:
        """Text files don't have thumbnails."""
        return None
