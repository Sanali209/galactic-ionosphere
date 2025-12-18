"""
Unit tests for ActionRegistry (Phase 2).
Testing action registration, retrieval, and triggering.
"""
import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QWidget
from src.ui.menus.action_registry import ActionRegistry

def test_action_registration(qapp):
    """Test registering an action."""
    if qapp is None: pytest.skip("No QApp")
    
    parent = QWidget()
    registry = ActionRegistry(main_window)
    
    callback = MagicMock()
    action = registry.register_action(
        "test_action", "Test Action", callback, "Ctrl+T", "Test tooltip")
    
    assert action is not None
    assert action.text() == "Test Action"
    assert action.shortcut().toString() == "Ctrl+T"
    assert action.statusTip() == "Test tooltip"

def test_action_retrieval():
    """Test getting action by name."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    callback = MagicMock()
    registry.register_action("my_action", "My Action", callback)
    
    action = registry.get_action("my_action")
    assert action is not None
    assert action.text() == "My Action"
    
    # Non-existent action
    assert registry.get_action("nonexistent") is None

def test_action_trigger():
    """Test that action triggers callback."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    callback = MagicMock()
    action = registry.register_action("test", "Test", callback)
    
    # Trigger action
    action.trigger()
    
    # Callback should be called
    callback.assert_called_once()

def test_action_context_enabled():
    """Test enabling action only in certain context."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    action = registry.register_action("save", "Save", lambda: None)
    
    # Simulate: save enabled only when document open
    registry.set_enabled("save", False)  # No doc open
    assert not action.isEnabled()
    
    registry.set_enabled("save", True)   # Doc open
    assert action.isEnabled()

def test_checkable_action():
    """Test checkable actions."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    action = registry.register_action(
        "toggle_panel", "Panel", lambda: None, checkable=True)
    
    assert action.isCheckable()
    
    # Set checked
    registry.set_checked("toggle_panel", True)
    assert action.isChecked()
    
    registry.set_checked("toggle_panel", False)
    assert not action.isChecked()

def test_update_action_text():
    """Test updating action text dynamically."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    action = registry.register_action("dynamic", "Original", lambda: None)
    assert action.text() == "Original"
    
    registry.update_text("dynamic", "Updated")
    assert action.text() == "Updated"
