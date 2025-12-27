"""
Tests for Foundation Enhancement components.
"""
import pytest
import asyncio
from typing import List

# Import new modules - NOTE: MessageBusSystem removed, now using EventBus
from src.core.events import EventBus, Events  # Events constants moved to events module
from src.core.lifecycle import AppState, LifecycleManager, LifecycleError
from src.core.decorators import system, on_lifecycle, subscribe_event
from src.core.plugins import Plugin, PluginState, PluginManager
from src.ucorefs.ai.driver_registry import AIDriver, DriverRegistry


# ============================================================
# EventBus Tests (Replaces MessageBusSystem Tests)
# ============================================================

class TestEventBus:
    """Tests for EventBus (unified pub-sub system)."""
    
    @pytest.fixture
    async def bus(self):
        """Create initialized event bus."""
        from unittest.mock import MagicMock
        bus = EventBus(MagicMock(), MagicMock())
        await bus.initialize()
        return bus
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus):
        """Test basic pub-sub."""
        received = []
        
        def handler(data):
            received.append(data)
        
        bus.subscribe("test.event", handler)
        bus.publish("test.event", {"value": 42})
        
        assert len(received) == 1
        assert received[0]["value"] == 42
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        """Test unsubscribe."""
        received = []
        
        def handler(data):
            received.append(data)
        
        bus.subscribe("test.event", handler)
        bus.unsubscribe("test.event", handler)
        bus.publish("test.event", {"value": 42})
        
        assert len(received) == 0
    
    @pytest.mark.asyncio
    async def test_async_publish(self, bus):
        """Test async publish."""
        received = []
        
        async def async_handler(data):
            received.append(data)
        
        bus.subscribe("test.event", async_handler)
        await bus.publish_async("test.event", {"value": 99})
        
        assert len(received) == 1
        assert received[0]["value"] == 99
    
    @pytest.mark.asyncio
    async def test_shutdown_clears(self, bus):
        """Test shutdown clears subscribers."""
        bus.subscribe("test.event", lambda d: None)
        await bus.shutdown()
        # After shutdown, publishing should not fail
        bus.publish("test.event", {})


# ============================================================
# LifecycleManager Tests
# ============================================================

class TestLifecycleManager:
    """Tests for LifecycleManager."""
    
    def test_initial_state(self):
        """Test initial state is CREATED."""
        lm = LifecycleManager()
        assert lm.state == AppState.CREATED
    
    def test_valid_transition(self):
        """Test valid state transition."""
        lm = LifecycleManager()
        lm.transition_to(AppState.INITIALIZED)
        assert lm.state == AppState.INITIALIZED
    
    def test_invalid_transition_raises(self):
        """Test invalid transition raises error."""
        lm = LifecycleManager()
        with pytest.raises(LifecycleError):
            lm.transition_to(AppState.RUNNING)  # Can't go CREATED -> RUNNING
    
    def test_hooks_called(self):
        """Test hooks are called on transition."""
        lm = LifecycleManager()
        called = []
        
        lm.register_hook(AppState.INITIALIZED, lambda: called.append("init"))
        lm.transition_to(AppState.INITIALIZED)
        
        assert "init" in called
    
    def test_listeners_notified(self):
        """Test listeners are notified."""
        lm = LifecycleManager()
        transitions = []
        
        lm.add_listener(lambda old, new: transitions.append((old, new)))
        lm.transition_to(AppState.INITIALIZED)
        
        assert transitions == [(AppState.CREATED, AppState.INITIALIZED)]


# ============================================================
# Decorator Tests
# ============================================================

class TestDecorators:
    """Tests for decorators."""
    
    def test_system_decorator(self):
        """Test @system decorator."""
        @system(depends_on=["ServiceA"], name="MySystem")
        class MySystem:
            pass
        
        assert MySystem.depends_on == ["ServiceA"]
        assert MySystem._system_name == "MySystem"
    
    def test_on_lifecycle_decorator(self):
        """Test @on_lifecycle decorator."""
        @on_lifecycle("STARTED")
        def my_hook():
            pass
        
        assert my_hook._lifecycle_state == "STARTED"
    
    def test_subscribe_event_decorator(self):
        """Test @subscribe_event decorator."""
        @subscribe_event("file.created", "file.deleted")
        def my_handler(**data):
            pass
        
        assert my_handler._subscribed_events == ["file.created", "file.deleted"]


# ============================================================
# Plugin Tests
# ============================================================

class TestPlugin:
    """Tests for Plugin system."""
    
    def test_plugin_lifecycle(self):
        """Test plugin lifecycle transitions."""
        class TestPlugin(Plugin):
            pass
        
        plugin = TestPlugin("test")
        assert plugin.state == PluginState.UNLOADED
        
        # Simulate load
        from unittest.mock import MagicMock
        plugin.load(MagicMock())
        assert plugin.state == PluginState.LOADED
        
        plugin.enable()
        assert plugin.state == PluginState.ENABLED
        
        plugin.start()
        assert plugin.state == PluginState.STARTED
        
        plugin.stop()
        assert plugin.state == PluginState.ENABLED


# ============================================================
# DriverRegistry Tests
# ============================================================

class TestDriverRegistry:
    """Tests for Driver Registry."""
    
    def test_driver_realizations(self):
        """Test driver realization management."""
        class TestDriver(AIDriver):
            def on_load(self):
                self.register_realization("gpu", "GPU_IMPL", tags=["fast"])
                self.register_realization("cpu", "CPU_IMPL", tags=["slow"])
        
        driver = TestDriver("test_driver")
        driver.load()
        
        assert driver.get_realization("gpu") == "GPU_IMPL"
        assert driver.get_realization("cpu") == "CPU_IMPL"
        assert len(driver.get_realization_by_tag("fast")) == 1
    
    def test_driver_active_realization(self):
        """Test setting active realization."""
        class TestDriver(AIDriver):
            def on_load(self):
                self.register_realization("v1", "V1_IMPL")
                self.register_realization("v2", "V2_IMPL")
        
        driver = TestDriver()
        driver.load()
        
        driver.set_active("v2")
        assert driver.get_active() == "V2_IMPL"
