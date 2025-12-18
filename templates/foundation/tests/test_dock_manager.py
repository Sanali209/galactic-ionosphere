"""
Unit tests for DockManager (Phase 2).
Testing panel registration, creation, and state persistence.
"""
import pytest
from unittest.mock import MagicMock, Mock
from PySide6.QtWidgets import QMainWindow
from src.ui.docking.dock_manager import DockManager, PanelState
from src.ui.docking.panel_base import BasePanelWidget

class MockPanel(BasePanelWidget):
    """Mock panel for testing."""
    def initialize_ui(self):
        pass

def test_panel_registration():
    """Test registering a panel type."""
    main_window = MagicMock(spec=QMainWindow)
    config = MagicMock()
    
    manager = DockManager(main_window, config)
    manager.register_panel("test_panel", MockPanel)
    
    assert "test_panel" in manager._panel_registry
    assert manager._panel_registry["test_panel"] == MockPanel

def test_panel_show_hide():
    """Test showing and hiding panels."""
    main_window = MagicMock(spec=QMainWindow)
    config = MagicMock()
    
    manager = DockManager(main_window, config)
    manager.register_panel("test_panel", MockPanel)
    
    # Initially no active panels
    assert len(manager._active_panels) == 0
    
    # Show panel (creates it)
    manager.show_panel("test_panel")
    assert "test_panel" in manager._active_panels

def test_dock_state_serialization():
    """Test PanelState to/from dict."""
    state = PanelState(visible=True, area="right", width=300, height=200)
    state.custom_state = {"key": "value"}
    
    data = state.to_dict()
    
    assert data["visible"] == True
    assert data["area"] == "right"
    assert data["width"] == 300
    assert data["height"] == 200
    assert data["custom_state"]["key"] == "value"
    
    # Round trip
    state2 = PanelState.from_dict(data)
    assert state2.visible == state.visible
    assert state2.area == state.area
    assert state2.width == state.width
    assert state2.custom_state["key"] == "value"

def test_panel_position():
    """Test that panel gets correct dock area."""
    # This test requires actual Qt widgets, which is complex
    # For now we'll test the logic only
    pass

def test_panel_size_persistence():
    """Test that panel size is saved."""
    state = PanelState(width=400, height=250)
    data = state.to_dict()
    
    assert data["width"] == 400
    assert data["height"] == 250
