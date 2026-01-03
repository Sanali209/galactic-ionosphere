import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.events import Signal
from src.core.config import ConfigManager
from src.core.commands.bus import CommandBus
from src.core.commands.base import ICommand, ICommandHandler

# --- Events Tests ---
def test_event_subscription():
    event = Signal("test_event")
    callback = MagicMock()
    
    event.connect(callback) # Fixed: connect
    event.emit("arg1", 123)
    
    callback.assert_called_with("arg1", 123)

def test_event_unsubscribe():
    event = Signal("test_event")
    callback = MagicMock()
    
    event.connect(callback) # Fixed: connect
    event.disconnect(callback) # Fixed: disconnect
    event.emit("arg1")
    
    callback.assert_not_called()

# --- Config Tests ---
def test_config_manager_defaults():
    config = ConfigManager("non_existent_config.json") # Use dummy file to avoid loading real one
    # Assuming default structure from implementation
    assert config.data.mongo.host == "localhost" # Fixed: defaults to localhost

def test_config_reactivity():
    config = ConfigManager("non_existent_config.json")
    observer = MagicMock()
    config.on_changed.connect(observer) # Fixed: connect
    
    config.update("general", "theme", "blue")
    
    assert config.data.general.theme == "blue"
    observer.assert_called()

# --- Command Bus Tests ---
@pytest.mark.asyncio
async def test_command_bus():
    bus = CommandBus(MagicMock(), MagicMock())
    
    class TestCommand(ICommand):
        pass
    
    handler = AsyncMock(spec=ICommandHandler)
    handler.handle.return_value = "handled"
    
    bus.register(TestCommand, handler)
    
    await bus.dispatch(TestCommand()) # Fixed: does not return result
    
    handler.handle.assert_called_once()

@pytest.mark.asyncio
async def test_command_bus_unregistered():
    bus = CommandBus(MagicMock(), MagicMock())
    class UnknownCommand(ICommand): pass
    
    with pytest.raises(ValueError):
        await bus.dispatch(UnknownCommand())
@pytest.mark.asyncio
async def test_task_system_sync_handler():
    from src.core.tasks.system import TaskSystem
    from src.core.locator import ServiceLocator
    from unittest.mock import patch
    import time

    # Mock locator and config
    locator = MagicMock()
    config = MagicMock()
    config.data.general.task_workers = 1

    # Setup System
    task_sys = TaskSystem(locator, config)
    # Mock TaskRecord to avoid DB
    with patch('src.core.tasks.system.TaskRecord') as MockRecord:
        # Mock instance
        mock_record_instance = AsyncMock()
        mock_record_instance.id = "task_123"
        mock_record_instance.status = "pending"
        mock_record_instance.handler_name = "sync_sleep"
        mock_record_instance.task_args = ["0.1"] # sleep 100ms
        
        # When saving, update status locally to simulate DB
        async def save_side_effect():
            if mock_record_instance.status == "running":
                pass
            return True
        mock_record_instance.save.side_effect = save_side_effect
        
        # When get is called, return our mock
        MockRecord.get = AsyncMock(return_value=mock_record_instance)
        MockRecord.find = AsyncMock(return_value=[]) # No recovery tasks
        
        # Initialize
        await task_sys.initialize()
        
        # Define blocking handler
        def blocking_handler(seconds: str):
            time.sleep(float(seconds))
            return "slept"
            
        task_sys.register_handler("sync_sleep", blocking_handler)
        
        # Trigger manually
        await task_sys._queue.put((1, "task_123"))
        
        # Wait for worker to pick up (allow loop switch)
        await asyncio.sleep(0.5) 
        
        # Verify
        assert mock_record_instance.status == "completed"
        assert mock_record_instance.result == "slept"
        
        await task_sys.shutdown()
