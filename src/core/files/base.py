from abc import ABC, abstractmethod
from typing import Dict, Any, List

class FileHandler(ABC):
    """
    Strategy interface for handling different file types.
    """
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of extensions this handler supports (e.g. ['.jpg', '.jpeg'])"""
        pass

    @abstractmethod
    async def extract_metadata(self, path: str) -> Dict[str, Any]:
        """Extracts XMP/Exif or other metadata."""
        pass

    @abstractmethod
    async def write_metadata(self, path: str, metadata: Dict[str, Any]) -> None:
        """Writes metadata to the file."""
        pass
        
    @abstractmethod
    async def generate_thumbnail(self, source_path: str, target_path: str, size: tuple = (256, 256)):
        """Generates a thumbnail image."""
        pass
        
    @abstractmethod
    async def get_dimensions(self, path: str) -> Dict[str, int]:
        """Returns {width, height}."""
        pass
