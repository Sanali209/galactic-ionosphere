
import sys
from unittest.mock import MagicMock, patch

# Mock PySide6QtAds to unblock imports if missing
sys.modules["PySide6QtAds"] = MagicMock()

import pytest
import asyncio

# We will need to import these after they are created/modified
# explicit imports to verify structure
from src.core.events import Signal, EventBus
from src.core.database.orm import IntField, StringField, CollectionRecord, Field
from src.core.service_decorator import Service
from src.core.base_system import BaseSystem

# --- Event System Tests ---

def test_signal_event():
    """Verify Signal (ObserverEvent) behavior."""
    sig = Signal("test_signal")
    mock_handler = MagicMock()
    
    sig.connect(mock_handler)
    sig.emit("data", 123)
    
    mock_handler.assert_called_once_with("data", 123)
    
    sig.disconnect(mock_handler)
    sig.emit("data2")
    assert mock_handler.call_count == 1

@pytest.mark.asyncio
async def test_event_bus():
    """Verify EventBus behavior."""
    bus = EventBus(None, None)
    await bus.initialize()
    
    mock_handler = MagicMock()
    async_mock_handler = MagicMock()
    
    async def async_wrapper(data):
        async_mock_handler(data)
        
    bus.subscribe("test.event", mock_handler)
    bus.subscribe("test.async", async_wrapper)
    
    await bus.publish("test.event", "payload")
    mock_handler.assert_called_with("payload")
    
    await bus.publish("test.async", "async_payload")
    async_mock_handler.assert_called_with("async_payload")
    
    await bus.shutdown()

# --- ORM Tests ---

class TestModel(CollectionRecord):
    _collection_name = "tests"
    name = StringField()
    age = IntField()

def test_orm_validation_int():
    """Test IntField validation."""
    model = TestModel()
    model.age = 10
    assert model.age == 10
    
    # Should cast compatible types
    model.age = "20"
    assert model.age == 20
    assert isinstance(model.age, int)
    
    # Should fail incompatible types
    with pytest.raises((TypeError, ValueError)):
        model.age = "not an int"

def test_orm_validation_string():
    """Test StringField validation."""
    model = TestModel()
    model.name = "test"
    assert model.name == "test"
    
    model.name = 123
    assert model.name == "123" # Should cast to string

# --- Service Decorator Tests ---

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
        
        # Verify logger was called
        # Note: self._is_ready is NOT set automatically by decorator unless we enforce it.
        # But logging should happen.
        assert mock_log.called
        
        # Shutdown logging
        await svc.shutdown()
        assert mock_log.call_count >= 2

@pytest.mark.asyncio
async def test_event_bus():
    """Verify EventBus behavior."""
    bus = EventBus(None, None)
    await bus.initialize()
    
    mock_handler = MagicMock()
    mock_handler.__name__ = "mock_handler" # Fix AttributeError
    
    async_mock_handler = MagicMock()
    async_mock_handler.__name__ = "async_mock_handler"
    
    async def async_wrapper(data):
        async_mock_handler(data)
        
    bus.subscribe("test.event", mock_handler)
    bus.subscribe("test.async", async_wrapper)
    
    await bus.publish("test.event", "payload")
    mock_handler.assert_called_with("payload")
    
    await bus.publish("test.async", "async_payload")
    async_mock_handler.assert_called_with("async_payload")
    
    await bus.shutdown()

