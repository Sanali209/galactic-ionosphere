"""
Dockable Relations Panel for UExplorer.

Works with DockingService (QWidget-based).
"""
from typing import TYPE_CHECKING, Optional
from PySide6.QtWidgets import QVBoxLayout, QWidget

import sys
from pathlib import Path
from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.relation_panel import RelationTreeWidget

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator

class RelationsPanel(PanelBase):
    """Dockable relations panel."""
    
    def __init__(self, parent: Optional[QWidget], locator: "ServiceLocator") -> None:
        self._tree: Optional[RelationTreeWidget] = None
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
