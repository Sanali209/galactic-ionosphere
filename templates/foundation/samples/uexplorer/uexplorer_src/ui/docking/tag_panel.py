"""
Dockable Tag Panel for UExplorer.

Works with DockingService (QWidget-based).
"""
from PySide6.QtWidgets import QVBoxLayout

import sys
from pathlib import Path
widgets_path = Path(__file__).parent.parent / "widgets"
if str(widgets_path) not in sys.path:
    sys.path.insert(0, str(widgets_path))

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.tag_tree import TagTreeWidget


class TagPanel(PanelBase):
    """
    Tag browser panel.
    
    Shows hierarchical tag structure and allows:
    - Browsing tags
    - Filtering by tags
    - Drag & drop files onto tags
    """
    
    def __init__(self, parent, locator):
        """
        Args:
            parent: Parent widget
            locator: ServiceLocator for services
        """
        self._tree = None
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._tree = TagTreeWidget(self.locator)
        layout.addWidget(self._tree)
    
    @property
    def tree(self) -> TagTreeWidget:
        return self._tree
    
    def on_update(self, context=None):
        """Refresh tags when panel updated."""
        if self._tree:
            import asyncio
            asyncio.ensure_future(self._tree.refresh_tags())
    
    def get_state(self) -> dict:
        """Save panel state for persistence."""
        state = {}
        if self._tree and hasattr(self._tree, 'model'):
            try:
                expanded_items = self._tree.get_expanded_items() if hasattr(self._tree, 'get_expanded_items') else []
                if expanded_items:
                    state['expanded_tags'] = expanded_items
            except:
                pass
        return state
    
    def set_state(self, state: dict):
        """Restore panel state from saved data."""
        if not state or not self._tree:
            return
        
        if 'expanded_tags' in state and hasattr(self._tree, 'expand_items'):
            try:
                self._tree.expand_items(state['expanded_tags'])
            except:
                pass
