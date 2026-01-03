
import pytest
from unittest.mock import MagicMock
from src.core.base_system import BaseSystem
from src.core.events import EventBus
from src.core.decorators import subscribe_event

class MockService(BaseSystem):
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self.event_called = False
        
    @subscribe_event("test_event")
    async def on_test_event(self, data):
        self.event_called = True

    async def initialize(self):
        await super().initialize()
        
    async def shutdown(self):
        pass

@pytest.mark.asyncio
async def test_auto_subscription():
    # Setup
    locator = MagicMock()
    config = MagicMock()
    event_bus = MagicMock()
    locator.get_system.side_effect = lambda cls: event_bus if cls == EventBus else None
    
    # Initialize service
    service = MockService(locator, config)
    await service.initialize()
    
    # Verify subscription
    event_bus.subscribe.assert_called_once()
    args = event_bus.subscribe.call_args[0]
    assert args[0] == "test_event"
    assert args[1] == service.on_test_event
