"""
EventBus - Comprehensive Unit Tests

Tests for the unified event bus system covering:
- Subscription/unsubscription
- Event publishing (sync/async)
- Multiple subscribers
- Error handling
- Edge cases
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.events.bus import EventBus


class TestEventBusBasic:
    """Basic EventBus functionality tests."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a test EventBus instance."""
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    def test_event_bus_initialization(self, event_bus):
        """Test EventBus initializes with empty subscribers."""
        assert hasattr(event_bus, '_subscribers')
        assert isinstance(event_bus._subscribers, dict)
        assert len(event_bus._subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, event_bus):
        """Test EventBus lifecycle methods."""
        await event_bus.initialize()
        assert event_bus.is_ready is True
        
        await event_bus.shutdown()
        assert len(event_bus._subscribers) == 0


class TestEventBusSubscription:
    """Test event subscription functionality."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    def test_subscribe_to_event(self, event_bus):
        """Test subscribing to an event."""
        def handler(data):
            pass
        
        event_bus.subscribe("test.event", handler)
        
        assert "test.event" in event_bus._subscribers
        assert handler in event_bus._subscribers["test.event"]
    
    def test_subscribe_multiple_handlers(self, event_bus):
        """Test subscribing multiple handlers to same event."""
        def handler1(data):
            pass
        
        def handler2(data):
            pass
        
        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)
        
        assert len(event_bus._subscribers["test.event"]) == 2
        assert handler1 in event_bus._subscribers["test.event"]
        assert handler2 in event_bus._subscribers["test.event"]
    
    def test_subscribe_duplicate_handler(self, event_bus):
        """Test subscribing same handler twice doesn't duplicate."""
        def handler(data):
            pass
        
        event_bus.subscribe("test.event", handler)
        event_bus.subscribe("test.event", handler)  # Duplicate
        
        # Should only be added once
        assert len(event_bus._subscribers["test.event"]) == 1
    
    def test_subscribe_multiple_events(self, event_bus):
        """Test subscribing to multiple different events."""
        def handler(data):
            pass
        
        event_bus.subscribe("event.one", handler)
        event_bus.subscribe("event.two", handler)
        
        assert len(event_bus._subscribers) == 2
        assert handler in event_bus._subscribers["event.one"]
        assert handler in event_bus._subscribers["event.two"]


class TestEventBusUnsubscription:
    """Test event unsubscription functionality."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    def test_unsubscribe_from_event(self, event_bus):
        """Test unsubscribing from an event."""
        def handler(data):
            pass
        
        event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe("test.event", handler)
        
        assert handler not in event_bus._subscribers.get("test.event", [])
    
    def test_unsubscribe_nonexistent_event(self, event_bus):
        """Test unsubscribing from non-existent event doesn't error."""
        def handler(data):
            pass
        
        # Should not raise exception
        event_bus.unsubscribe("nonexistent.event", handler)
    
    def test_unsubscribe_nonexistent_handler(self, event_bus):
        """Test unsubscribing non-subscribed handler doesn't error."""
        def handler1(data):
            pass
        
        def handler2(data):
            pass
        
        event_bus.subscribe("test.event", handler1)
        
        # Should not raise exception
        event_bus.unsubscribe("test.event", handler2)
        
        # handler1 should still be subscribed
        assert handler1 in event_bus._subscribers["test.event"]


class TestEventBusPublishing:
    """Test event publishing functionality."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_publish_to_sync_handler(self, event_bus):
        """Test publishing to synchronous handler."""
        received_data = []
        
        def handler(data):
            received_data.append(data)
        
        event_bus.subscribe("test.event", handler)
        await event_bus.publish("test.event", {"value": 42})
        
        assert len(received_data) == 1
        assert received_data[0]["value"] == 42
    
    @pytest.mark.asyncio
    async def test_publish_to_async_handler(self, event_bus):
        """Test publishing to asynchronous handler."""
        received_data = []
        
        async def handler(data):
            received_data.append(data)
        
        event_bus.subscribe("test.event", handler)
        await event_bus.publish("test.event", {"value": 99})
        
        assert len(received_data) == 1
        assert received_data[0]["value"] == 99
    
    @pytest.mark.asyncio
    async def test_publish_to_multiple_handlers(self, event_bus):
        """Test publishing to multiple handlers."""
        call_order = []
        
        def handler1(data):
            call_order.append(1)
        
        def handler2(data):
            call_order.append(2)
        
        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)
        
        await event_bus.publish("test.event", None)
        
        assert len(call_order) == 2
        assert 1 in call_order
        assert 2 in call_order
    
    @pytest.mark.asyncio
    async def test_publish_without_data(self, event_bus):
        """Test publishing without data."""
        received = []
        
        def handler(data):
            received.append(data)
        
        event_bus.subscribe("test.event", handler)
        await event_bus.publish("test.event")
        
        assert len(received) == 1
        assert received[0] is None
    
    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_event(self, event_bus):
        """Test publishing to event with no subscribers."""
        # Should not raise exception
        await event_bus.publish("nonexistent.event", {"data": "test"})


class TestEventBusErrorHandling:
    """Test error handling in event bus."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_stop_publishing(self, event_bus):
        """Test that exception in one handler doesn't stop others."""
        call_log = []
        
        def failing_handler(data):
            raise ValueError("Test error")
        
        def working_handler(data):
            call_log.append("called")
        
        event_bus.subscribe("test.event", failing_handler)
        event_bus.subscribe("test.event", working_handler)
        
        # Should not raise exception
        await event_bus.publish("test.event", None)
        
        # Working handler should still be called
        assert len(call_log) == 1
    
    @pytest.mark.asyncio
    async def test_async_handler_exception_handling(self, event_bus):
        """Test async handler exceptions are caught."""
        async def failing_handler(data):
            raise RuntimeError("Async test error")
        
        event_bus.subscribe("test.event", failing_handler)
        
        # Should not raise exception
        await event_bus.publish("test.event", None)


class TestEventBusSyncPublishing:
    """Test synchronous publishing functionality."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    def test_publish_sync_to_sync_handler(self, event_bus):
        """Test synchronous publish to sync handler."""
        received_data = []
        
        def handler(data):
            received_data.append(data)
        
        event_bus.subscribe("test.event", handler)
        event_bus.publish_sync("test.event", {"value": 123})
        
        assert len(received_data) == 1
        assert received_data[0]["value"] == 123
    
    def test_publish_sync_to_async_handler(self, event_bus):
        """Test synchronous publish creates task for async handler."""
        async def handler(data):
            pass
        
        event_bus.subscribe("test.event", handler)
        
        # Should not raise exception (async handler will run as task)
        event_bus.publish_sync("test.event", None)
    
    def test_publish_sync_error_handling(self, event_bus):
        """Test sync publish handles errors."""
        def failing_handler(data):
            raise ValueError("Sync test error")
        
        event_bus.subscribe("test.event", failing_handler)
        
        # Should not raise exception
        event_bus.publish_sync("test.event", None)


class TestEventBusIntegration:
    """Integration tests for EventBus."""
    
    @pytest.fixture
    def event_bus(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return EventBus(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_subscribe_publish_unsubscribe_flow(self, event_bus):
        """Test complete subscribe -> publish -> unsubscribe flow."""
        received = []
        
        def handler(data):
            received.append(data)
        
        # Subscribe
        event_bus.subscribe("test.event", handler)
        
        # Publish - should receive
        await event_bus.publish("test.event", "first")
        assert len(received) == 1
        
        # Unsubscribe
        event_bus.unsubscribe("test.event", handler)
        
        # Publish - should not receive
        await event_bus.publish("test.event", "second")
        assert len(received) == 1  # Still only 1
    
    @pytest.mark.asyncio
    async def test_multiple_events_independent(self, event_bus):
        """Test multiple events operate independently."""
        event1_data = []
        event2_data = []
        
        def handler1(data):
            event1_data.append(data)
        
        def handler2(data):
            event2_data.append(data)
        
        event_bus.subscribe("event.one", handler1)
        event_bus.subscribe("event.two", handler2)
        
        await event_bus.publish("event.one", "first")
        await event_bus.publish("event.two", "second")
        
        assert len(event1_data) == 1
        assert len(event2_data) == 1
        assert event1_data[0] == "first"
        assert event2_data[0] == "second"
    
    @pytest.mark.asyncio
    async def test_shutdown_clears_subscribers(self, event_bus):
        """Test shutdown clears all subscribers."""
        def handler(data):
            pass
        
        event_bus.subscribe("event.one", handler)
        event_bus.subscribe("event.two", handler)
        
        assert len(event_bus._subscribers) == 2
        
        await event_bus.shutdown()
        
        assert len(event_bus._subscribers) == 0
