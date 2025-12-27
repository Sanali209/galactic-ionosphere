import pytest
from src.core.locator import ServiceLocator
from src.core.config import ConfigManager

def test_locator_singleton():
    sl1 = ServiceLocator()
    sl2 = ServiceLocator()
    assert sl1 is sl2

def test_config_load():
    # Ensure it doesn't crash on default generic path
    config = ConfigManager("test_config.json")
    assert config.data.general.debug_mode is True
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
