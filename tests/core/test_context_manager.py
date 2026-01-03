
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.base_system import BaseSystem

class ContextService(BaseSystem):
    async def initialize(self):
        await super().initialize()
        
    async def shutdown(self):
        await super().shutdown()

@pytest.mark.asyncio
async def test_async_context_manager():
    # Setup
    locator = MagicMock()
    config = MagicMock()
    
    # Mock auto_subscribe_events to avoid dependency on EventBus
    with patch.object(ContextService, '_auto_subscribe_events'):
        service = ContextService(locator, config)
        
        # Verify not ready initially
        assert not service.is_ready
        
        # Test Context Entry
        async with service as s:
            assert s is service
            assert service.is_ready
            
        # Test Context Exit
        assert not service.is_ready
