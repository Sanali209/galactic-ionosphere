"""
Tests for LibraryDialog.
"""
import pytest
from PySide6.QtWidgets import QListWidget, QPushButton
import sys
from pathlib import Path

# Add paths
uexplorer_path = Path(__file__).parent.parent
sys.path.insert(0, str(uexplorer_path / "src" / "ui" / "dialogs"))

from library_dialog import LibraryDialog


class TestLibraryDialog:
    """Test LibraryDialog functionality."""
    
    @pytest.mark.asyncio
    async def test_dialog_initialization(self, qtbot, locator):
        """Test LibraryDialog initializes correctly."""
        dialog = LibraryDialog(locator)
        qtbot.addWidget(dialog)
        
        assert dialog is not None
        assert dialog.locator == locator
        assert isinstance(dialog.root_list, QListWidget)
    
    @pytest.mark.asyncio
    async def test_buttons_exist(self, qtbot, locator):
        """Test add and remove buttons exist."""
        dialog = LibraryDialog(locator)
        qtbot.addWidget(dialog)
        
        assert dialog.add_btn is not None
        assert dialog.remove_btn is not None
        assert isinstance(dialog.add_btn, QPushButton)
        assert isinstance(dialog.remove_btn, QPushButton)
    
    @pytest.mark.asyncio
    async def test_dialog_title(self, qtbot, locator):
        """Test dialog has correct title."""
        dialog = LibraryDialog(locator)
        qtbot.addWidget(dialog)
        
        assert "Library" in dialog.windowTitle()
    
    @pytest.mark.asyncio
    async def test_remove_button_disabled_when_no_selection(self, qtbot, locator):
        """Test remove button disabled when nothing selected."""
        dialog = LibraryDialog(locator)
        qtbot.addWidget(dialog)
        
        # Clear selection
        dialog.root_list.clearSelection()
        dialog.on_selection_changed()
        
        # Remove button should be disabled
        assert dialog.remove_btn.isEnabled() == False
