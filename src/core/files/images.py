import os
from typing import Dict, Any, List
# Placeholder for PIL/PyExiv2 imports, assuming installed
# from PIL import Image
# import pyexiv2

from src.core.files.base import FileHandler

class JpgHandler(FileHandler):
    @property
    def supported_extensions(self) -> List[str]:
        return ['.jpg', '.jpeg']

    async def extract_metadata(self, path: str) -> Dict[str, Any]:
        # Minimal stub for now until Phase 2
        return {"stub": "metadata"}

    async def generate_thumbnail(self, source_path: str, target_path: str, size: tuple = (256, 256)):
        # Stub
        pass

    async def get_dimensions(self, path: str) -> Dict[str, int]:
        # Stub
        return {"width": 0, "height": 0}

class PngHandler(FileHandler):
    @property
    def supported_extensions(self) -> List[str]:
        return ['.png']

    async def extract_metadata(self, path: str) -> Dict[str, Any]:
        return {}

    async def generate_thumbnail(self, source_path: str, target_path: str, size: tuple = (256, 256)):
        pass

    async def get_dimensions(self, path: str) -> Dict[str, int]:
        return {"width": 0, "height": 0}

class FileHandlerFactory:
    _handlers: Dict[str, FileHandler] = {}
    
    @classmethod
    def register(cls, handler: FileHandler):
        for ext in handler.supported_extensions:
            cls._handlers[ext.lower()] = handler
            
    @classmethod
    def get_handler(cls, ext: str) -> FileHandler:
        return cls._handlers.get(ext.lower())

# Auto-register default handlers?
FileHandlerFactory.register(JpgHandler())
FileHandlerFactory.register(PngHandler())
