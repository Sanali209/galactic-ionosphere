import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.core.service_decorator import Service
from src.core.base_system import BaseSystem

class MockLocator:
    def register_service(self, svc): pass

@Service
class MyService(BaseSystem):
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass

@pytest.mark.asyncio
async def test_service_decorator():
    """Verify @Service decorator handles lifecycle logging and state."""
    locator = MockLocator()
    svc = MyService(locator, None)
    
    # BaseSystem sets _is_ready=False initially
    assert not svc.is_ready
    
    with patch("loguru.logger.info") as mock_log:
        await svc.initialize()
        assert mock_log.called
        
        await svc.shutdown()
        assert mock_log.call_count >= 2
