
import sys
from unittest.mock import MagicMock

# Only mock problematic modules
sys.modules["PySide6QtAds"] = MagicMock()
sys.modules["qasync"] = MagicMock()
sys.modules["src.core.bootstrap"] = MagicMock()

import pytest
from PySide6.QtWidgets import QWidget, QMainWindow
from PySide6.QtGui import QAction
from src.ui.menus.action_registry import ActionRegistry

def test_action_registration(qapp):
    """Test registering an action."""
    main_window = QWidget()
    registry = ActionRegistry(main_window)
    
    callback = MagicMock()
    action = registry.register_action(
        "test_action", "Test Action", callback, "Ctrl+T", "Test tooltip")
    
    # Real QAction assertions
    assert isinstance(action, QAction)
    assert action.text() == "Test Action"
    assert action.shortcut().toString() == "Ctrl+T"
    assert action.statusTip() == "Test tooltip"

def test_action_retrieval(qapp):
    """Test getting action by name."""
    main_window = QWidget()
    registry = ActionRegistry(main_window)
    
    registry.register_action("my_action", "My Action", MagicMock())
    
    action = registry.get_action("my_action")
    assert isinstance(action, QAction)
    assert action.text() == "My Action"
    
    # Non-existent action
    assert registry.get_action("nonexistent") is None

def test_context_enabled(qapp):
    """Test enabling action."""
    main_window = QWidget()
    registry = ActionRegistry(main_window)
    
    action = registry.register_action("save", "Save", lambda: None)
    
    registry.set_enabled("save", False)
    assert not action.isEnabled()
    
    registry.set_enabled("save", True)
    assert action.isEnabled()

def test_update_action_text(qapp):
    """Test updating action text."""
    main_window = QWidget()
    registry = ActionRegistry(main_window)
    
    action = registry.register_action("dynamic", "Original", lambda: None)
    assert action.text() == "Original"
    
    registry.update_text("dynamic", "Updated")
    assert action.text() == "Updated"
