"""
UCoreFS Types Package

File type drivers and registry.
"""
from src.ucorefs.types.driver import IFileDriver
from src.ucorefs.types.registry import FileTypeRegistry, registry
from src.ucorefs.types.default import DefaultDriver
from src.ucorefs.types.image import ImageDriver
from src.ucorefs.types.text import TextDriver

# Register built-in drivers
registry.register(ImageDriver)
registry.register(TextDriver)

__all__ = [
    "IFileDriver",
    "FileTypeRegistry",
    "registry",
    "DefaultDriver",
    "ImageDriver",
    "TextDriver",
]
