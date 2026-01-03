import pytest
from unittest.mock import MagicMock
from src.core.events import Signal

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
