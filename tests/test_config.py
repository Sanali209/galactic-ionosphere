import pytest
from src.core.locator import sl

def test_config_read_default():
    # sl is already initialized by conftest
    assert sl.config.data.ai.provider_id == "clip_local"

def test_config_update_event():
    received = []
    
    def on_change(section, key, val):
        received.append((section, key, val))
        
    sl.config.on_changed.connect(on_change)
    
    # Update value
    sl.config.update("ai", "device", "cuda")
    
    assert sl.config.data.ai.device == "cuda"
    assert len(received) > 0
    assert received[-1] == ("ai", "device", "cuda")
