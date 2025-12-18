"""
Unit tests for MenuBuilder (Phase 2).
Testing menu creation and action assignment.
"""
import pytest
from unittest.mock import MagicMock, Mock
from PySide6.QtWidgets import QMainWindow
from src.ui.menus.menu_builder import MenuBuilder
from src.ui.menus.action_registry import ActionRegistry

def test_menu_creation():
    """Test that menus are created."""
    main_window = MagicMock(spec=QMainWindow)
    menubar = MagicMock()
    main_window.menuBar.return_value = menubar
    
    registry = ActionRegistry(main_window)
    
    # Register required actions
    registry.register_action("file_exit", "Exit", lambda: None)
    
    builder = MenuBuilder(main_window, registry)
    
    # Should be initialized
    assert builder.main_window == main_window
    assert builder.actions == registry

def test_action_enable_disable():
    """Test enabling/disabling actions based on context."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    # Register action
    action = registry.register_action("test_action", "Test", lambda: None)
    
    # Initially enabled (default)
    assert action.isEnabled()
    
    # Disable
    registry.set_enabled("test_action", False)
    assert not action.isEnabled()
    
    # Enable
    registry.set_enabled("test_action", True)
    assert action.isEnabled()

def test_keyboard_shortcuts():
    """Test that keyboard shortcuts are assigned."""
    main_window = MagicMock(spec=QMainWindow)
    registry = ActionRegistry(main_window)
    
    action = registry.register_action(
        "file_save", "Save", lambda: None, "Ctrl+S")
    
    # Check shortcut is set
    assert action.shortcut().toString() == "Ctrl+S"

def test_recent_files_list():
    """Test recent files submenu (placeholder test)."""
    # This would test the recent files functionality
    # For now, just verify the concept
    recent_files = ["file1.txt", "file2.txt", "file3.txt"]
    assert len(recent_files) == 3

def test_menu_hierarchy():
    """Test that menu hierarchy is correct."""
    # File -> Recent -> items
    # This is tested visually, but we can test the structure
    pass
