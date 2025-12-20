"""
Dockable Album Panel for UExplorer.

Works with DockingService (QWidget-based).
"""
from PySide6.QtWidgets import QVBoxLayout

import sys
from pathlib import Path
widgets_path = Path(__file__).parent.parent / "widgets"
if str(widgets_path) not in sys.path:
    sys.path.insert(0, str(widgets_path))

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.album_tree import AlbumTreeWidget


class AlbumPanel(PanelBase):
    """Dockable album tree panel."""
    
    def __init__(self, parent, locator):
        self._tree = None
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._tree = AlbumTreeWidget(self.locator)
        layout.addWidget(self._tree)
    
    @property
    def tree(self) -> AlbumTreeWidget:
        return self._tree
    
    def on_update(self, context=None):
        """Refresh albums when panel updated."""
        if self._tree:
            import asyncio
            asyncio.ensure_future(self._tree.refresh_albums())
