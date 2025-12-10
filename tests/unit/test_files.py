import pytest
from src.core.files.base import FileHandler
from src.core.files.images import FileHandlerFactory, JpgHandler, PngHandler

def test_handler_registration():
    # Factory should auto-register on import if code executes, 
    # but unit test isolation implies we might need to ensure imports or manual registration.
    
    # In test environment, imported module executes top-level code.
    h = FileHandlerFactory.get_handler(".jpg")
    assert isinstance(h, JpgHandler)
    
    h2 = FileHandlerFactory.get_handler(".png")
    assert isinstance(h2, PngHandler)
    
    h3 = FileHandlerFactory.get_handler(".unknown")
    assert h3 is None

def test_jpg_handler_properties():
    h = JpgHandler()
    assert ".jpg" in h.supported_extensions
    assert ".jpeg" in h.supported_extensions
