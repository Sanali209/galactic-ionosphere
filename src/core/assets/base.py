from abc import ABC, abstractmethod
from typing import Dict, Any, List

class IAssetHandler(ABC):
    """
    Strategy for handling specific asset types (e.g. Images, Videos, Virtual).
    """
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """e.g. ['image/jpeg', 'image/png'] or extensions ['.jpg']"""
        pass

    @abstractmethod
    async def process(self, path: str) -> Dict[str, Any]:
        """Process file and return metadata."""
        pass
