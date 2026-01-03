
import pytest
from src.ucorefs.extractors.protocols import ExtractorProtocol
from src.ucorefs.extractors.blip_extractor import BLIPExtractor
from src.ucorefs.extractors.base import Extractor

def test_instance_check():
    """Verify runtime instance checking."""
    # Create a dummy class that satisfies the protocol
    class ValidExtractor:
        name = "test"
        phase = 1
        priority = 10
        batch_supported = True
        
        async def extract(self, files): return {}
        async def store(self, file_id, result): return True
        def can_process(self, file): return True

    extractor = ValidExtractor()
    assert isinstance(extractor, ExtractorProtocol)

def test_blip_instance_check():
    """Verify BLIPExtractor instance satisfies protocol."""
    # We mock __init__ to avoid loading heavy models
    class MockBLIP(BLIPExtractor):
        def __init__(self):
            self._model = None
            
    extractor = MockBLIP()
    # Mock missing attributes if necessary, but BLIPExtractor should have them as class attrs
    # Protocol matching checks instance dict + class dict
    
    assert isinstance(extractor, ExtractorProtocol)
