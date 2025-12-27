"""
Tests for MainWindow.
"""
import pytest
from PySide6.QtWidgets import QStatusBar, QMenuBar
from pathlib import Path

# Add paths
uexplorer_path = Path(__file__).parent.parent
foundation_path = uexplorer_path.parent.parent / "templates" / "foundation"
from main_window import MainWindow

class TestMainWindow:
    """Test MainWindow functionality."""
    
    @pytest.mark.asyncio
    async def test_window_initialization(self, qtbot, locator):
        """Test MainWindow initializes correctly."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        assert window is not None
        assert window.locator == locator
    
    @pytest.mark.asyncio
    async def test_window_title(self, qtbot, locator):
        """Test window has correct title."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        assert "UExplorer" in window.windowTitle()
    
    @pytest.mark.asyncio
    async def test_menubar_exists(self, qtbot, locator):
        """Test menu bar exists."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        assert window.menuBar() is not None
        assert isinstance(window.menuBar(), QMenuBar)
    
    @pytest.mark.asyncio
    async def test_statusbar_exists(self, qtbot, locator):
        """Test status bar exists."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        assert window.statusBar() is not None
        assert isinstance(window.statusBar(), QStatusBar)
    
    @pytest.mark.asyncio
    async def test_menus_created(self, qtbot, locator):
        """Test File, Edit, View, Tools menus exist."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        menubar = window.menuBar()
        menu_titles = [action.text() for action in menubar.actions()]
        
        assert any("File" in title for title in menu_titles)
        assert any("Edit" in title for title in menu_titles)
        assert any("View" in title for title in menu_titles)
        assert any("Tools" in title for title in menu_titles)
    
    @pytest.mark.asyncio
    async def test_window_size(self, qtbot, locator):
        """Test window has reasonable default size."""
        window = MainWindow(locator)
        qtbot.addWidget(window)
        
        size = window.size()
        assert size.width() > 800
        assert size.height() > 600
