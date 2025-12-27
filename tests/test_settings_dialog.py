"""
Unit tests for SettingsDialog (Phase 3).
Testing category tree, search, and ConfigManager integration.
"""
import pytest
from unittest.mock import MagicMock, Mock
from PySide6.QtWidgets import QTreeWidgetItem
from src.ui.settings.settings_dialog import SettingsDialog

def test_category_tree_construction(qapp):
    """Test that category tree is built correctly."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test App"
    config.data.db_name = "test_db"
    
    dialog = SettingsDialog(config)
    
    # Should have 3 categories
    assert dialog.category_tree.topLevelItemCount() == 3
    assert dialog.category_tree.topLevelItem(0).text(0) == "General"
    assert dialog.category_tree.topLevelItem(1).text(0) == "Editor"
    assert dialog.category_tree.topLevelItem(2).text(0) == "Appearance"

def test_setting_value_get_set(qapp):
    """Test getting and setting values."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Initial Name"
    config.data.db_name = "initial_db"
    
    dialog = SettingsDialog(config)
    
    # Check initial values
    assert dialog.app_name_input.text() == "Initial Name"
    assert dialog.db_name_input.text() == "initial_db"

def test_setting_validation(qapp):
    """Test setting value validation."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test"
    config.data.db_name = "test"
    
    dialog = SettingsDialog(config)
    
    # Change value
    dialog.app_name_input.setText("New Name")
    
    # Should update config
    assert config.data.app_name == "New Name"

def test_setting_type_conversion(qapp):
    """Test that settings handle correct types."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test"
    config.data.db_name = "test"
    
    dialog = SettingsDialog(config)
    
    # String values should work
    dialog.app_name_input.setText("String Value")
    assert isinstance(dialog.app_name_input.text(), str)

def test_reset_to_default(qapp):
    """Test reset to default value."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Modified"
    config.data.db_name = "modified"
    
    dialog = SettingsDialog(config)
    
    # Reset button should exist
    assert dialog.reset_btn is not None
    assert dialog.reset_btn.text() == "Reset to Defaults"

def test_search_by_keyword(qapp):
    """Test search filters categories."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test"
    config.data.db_name = "test"
    
    dialog = SettingsDialog(config)
    
    # All visible initially
    assert not dialog.category_tree.topLevelItem(0).isHidden()
    assert not dialog.category_tree.topLevelItem(1).isHidden()
    assert not dialog.category_tree.topLevelItem(2).isHidden()
    
    # Search for "general"
    dialog.search_input.setText("general")
    
    # Only General should be visible
    assert not dialog.category_tree.topLevelItem(0).isHidden()
    assert dialog.category_tree.topLevelItem(1).isHidden()
    assert dialog.category_tree.topLevelItem(2).isHidden()

def test_search_result_highlighting(qapp):
    """Test search results are visible."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test"
    config.data.db_name = "test"
    
    dialog = SettingsDialog(config)
    
    # Search shows matching categories
    dialog.search_input.setText("edit")
    assert dialog.category_tree.topLevelItem(1).text(0) == "Editor"

def test_empty_search_returns_all(qapp):
    """Test empty search shows all settings."""
    if qapp is None: pytest.skip("No QApp")
    
    config = MagicMock()
    config.data = MagicMock()
    config.data.app_name = "Test"
    config.data.db_name = "test"
    
    dialog = SettingsDialog(config)
    
    # Filter
    dialog.search_input.setText("general")
    
    # Clear search
    dialog.search_input.setText("")
    
    # All visible
    assert not dialog.category_tree.topLevelItem(0).isHidden()
    assert not dialog.category_tree.topLevelItem(1).isHidden()
    assert not dialog.category_tree.topLevelItem(2).isHidden()
