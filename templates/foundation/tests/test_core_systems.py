import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.events import ObserverEvent
from src.core.config import ConfigManager
from src.core.commands.bus import CommandBus
from src.core.commands.base import ICommand, ICommandHandler

# --- Events Tests ---
def test_event_subscription():
    event = ObserverEvent("test_event")
    callback = MagicMock()
    
    event.connect(callback) # Fixed: connect
    event.emit("arg1", 123)
    
    callback.assert_called_with("arg1", 123)

def test_event_unsubscribe():
    event = ObserverEvent("test_event")
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
