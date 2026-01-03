
import sys
from unittest.mock import MagicMock

# Only mock problematic modules
sys.modules["PySide6QtAds"] = MagicMock()
sys.modules["qasync"] = MagicMock()
sys.modules["src.core.bootstrap"] = MagicMock()

import pytest
from PySide6.QtWidgets import QMainWindow
from src.ui.menus.menu_builder import MenuBuilder
from src.ui.menus.action_registry import ActionRegistry

def test_menu_creation(qapp):
    """Test menu creation."""
    """Test menu creation."""
    main_window = QMainWindow()
    menubar = MagicMock()
    # Mock menuBar() method of QMainWindow
    main_window.menuBar = MagicMock(return_value=menubar)
    
    # Use real registry but mocked window
    registry = ActionRegistry(main_window)
    registry.register_action("file_exit", "Exit", lambda: None)
    
    builder = MenuBuilder(main_window, registry)
    
    assert builder.main_window == main_window
    assert builder.actions == registry
    # builder.build() is called in __init__, which calls create_menus
    # We can check if menubar.addMenu was called
    assert menubar.addMenu.call_count > 0

def test_integration_with_real_actions(qapp):
    """Test builder with real actions."""
    main_window = QMainWindow()
    registry = ActionRegistry(main_window)
    registry.register_action("test", "Test", lambda: None)
    
    # Just verify it doesn't crash
    MenuBuilder(main_window, registry)
