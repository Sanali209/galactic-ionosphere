"""
Dockable Relations Panel for UExplorer.

Works with DockingService (QWidget-based).
"""
from PySide6.QtWidgets import QVBoxLayout

import sys
from pathlib import Path
from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.relation_panel import RelationTreeWidget

class RelationsPanel(PanelBase):
    """Dockable relations panel."""
    
    def __init__(self, parent, locator):
        self._tree = None
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._tree = RelationTreeWidget(self.locator)
        layout.addWidget(self._tree)
    
    @property
    def tree(self) -> RelationTreeWidget:
        return self._tree
