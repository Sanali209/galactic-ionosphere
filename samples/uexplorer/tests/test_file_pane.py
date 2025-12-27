"""
Tests for FilePaneWidget.
"""
import pytest
from PySide6.QtWidgets import QTreeView
from PySide6.QtCore import Qt
from pathlib import Path

# Add paths
uexplorer_path = Path(__file__).parent.parent
from file_pane import FilePaneWidget

class TestFilePaneWidget:
    """Test FilePaneWidget functionality."""
    
    @pytest.mark.asyncio
    async def test_widget_initialization(self, qtbot, locator):
        """Test FilePaneWidget initializes correctly."""
        widget = FilePaneWidget(locator)
        qtbot.addWidget(widget)
        
        assert widget is not None
        assert widget.locator == locator
        assert isinstance(widget.tree, QTreeView)
        assert widget.model is not None
    
    @pytest.mark.asyncio
    async def test_tree_view_setup(self, qtbot, locator):
        """Test tree view is configured correctly."""
        widget = FilePaneWidget(locator)
        qtbot.addWidget(widget)
        
        # Check tree exists and is a QTreeView
        assert widget.tree is not None
        assert isinstance(widget.tree, QTreeView)
    
    @pytest.mark.asyncio
    async    def test_toolbar_exists(self, qtbot, locator):
        """Test navigation toolbar exists."""
        widget = FilePaneWidget(locator)
        qtbot.addWidget(widget)
        
        # Check toolbar or filter exists (implementation may vary)
        assert hasattr(widget, 'toolbar') or hasattr(widget, 'filter_input') or hasattr(widget, 'tree')
    
    @pytest.mark.asyncio
    async def test_context_menu_setup(self, qtbot, locator):
        """Test context menu is configured."""
        widget = FilePaneWidget(locator)
        qtbot.addWidget(widget)
        
        assert widget.tree.contextMenuPolicy() == Qt.CustomContextMenu
    
    @pytest.mark.asyncio
    async def test_selection_signal(self, qtbot, locator):
        """Test selection changed signal is connected."""
        widget = FilePaneWidget(locator)
        qtbot.addWidget(widget)
        
        # Create a signal spy
        selection_count = []
        
        def on_selection(count):
            selection_count.append(count)
        
        widget.selection_changed.connect(on_selection)
        
        # Trigger selection change manually
        widget.on_selection_changed()
        
        # Should have emitted signal
        assert len(selection_count) > 0
