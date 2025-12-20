"""
Dockable Properties Panel (Metadata) for UExplorer.

Works with DockingService (QWidget-based).
"""
from PySide6.QtWidgets import QVBoxLayout

import sys
from pathlib import Path
widgets_path = Path(__file__).parent.parent / "widgets"
if str(widgets_path) not in sys.path:
    sys.path.insert(0, str(widgets_path))

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.metadata_panel import MetadataPanel


class PropertiesPanel(PanelBase):
    """Dockable properties/metadata panel."""
    
    def __init__(self, parent, locator):
        self._metadata = None
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._metadata = MetadataPanel(self.locator)
        layout.addWidget(self._metadata)
    
    @property
    def metadata_panel(self) -> MetadataPanel:
        return self._metadata
    
    def set_file(self, file_id: str):
        """Set current file to display."""
        if self._metadata:
            self._metadata.set_file(file_id)
