"""
Dockable Properties Panel (Metadata) for UExplorer.

Works with DockingService (QWidget-based).
"""
import asyncio
from PySide6.QtWidgets import QVBoxLayout
from bson import ObjectId
from loguru import logger

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
        """Set current file to display by fetching record from database."""
        if not self._metadata:
            return
        
        if not file_id:
            self._metadata.clear()
            return
        
        # Fetch FileRecord from database
        asyncio.ensure_future(self._load_file(file_id))
    
    async def _load_file(self, file_id: str):
        """Load FileRecord and update metadata panel."""
        try:
            from src.ucorefs.models import FileRecord
            
            obj_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            record = await FileRecord.get(obj_id)
            
            if record:
                self._metadata.set_file(record)
            else:
                logger.warning(f"FileRecord not found: {file_id}")
                self._metadata.clear()
        except Exception as e:
            logger.error(f"Failed to load file {file_id}: {e}")
            self._metadata.clear()

