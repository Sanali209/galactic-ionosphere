"""
Dockable Properties Panel (Metadata) for UExplorer.

Supports multiple modes: Properties, Detections.
Works with DockingService (QWidget-based).
"""
from typing import TYPE_CHECKING, Optional
import asyncio
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QStackedWidget, QWidget
)
from PySide6.QtCore import Qt
from bson import ObjectId
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.metadata_panel import MetadataPanel

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class PropertiesPanel(PanelBase):
    """
    Dockable properties/metadata panel with mode switching.
    
    Modes:
    - Properties: File metadata, rating, tags, description
    - Detections: Object detection results (YOLO, face detection)
    """
    
    MODE_PROPERTIES = 0
    MODE_DETECTIONS = 1
    
    def __init__(self, parent: Optional[QWidget], locator: "ServiceLocator") -> None:
        self._metadata: Optional[MetadataPanel] = None
        self._detections_widget: Optional["DetectionsWidget"] = None
        self._stack: Optional[QStackedWidget] = None
        self._mode_combo: Optional[QComboBox] = None
        self._current_file_id: Optional[str] = None
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI with mode selector and stacked content."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Mode selector header
        header = QHBoxLayout()
        
        header.addWidget(QLabel("Mode:"))
        
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("📋 Properties", self.MODE_PROPERTIES)
        self._mode_combo.addItem("🔍 Detections", self.MODE_DETECTIONS)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d40;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox:hover { border-color: #0d6efd; }
            QComboBox::drop-down { border: none; }
        """)
        header.addWidget(self._mode_combo)
        
        header.addStretch()
        
        layout.addLayout(header)
        
        # Stacked widget for different modes
        self._stack = QStackedWidget()
        
        # Properties mode (original MetadataPanel)
        self._metadata = MetadataPanel(self.locator)
        self._stack.addWidget(self._metadata)
        
        # Detections mode
        self._detections_widget = DetectionsWidget(self.locator)
        self._stack.addWidget(self._detections_widget)
        
        layout.addWidget(self._stack)
    
    def _on_mode_changed(self, index: int):
        """Handle mode combo change."""
        mode = self._mode_combo.itemData(index)
        self._stack.setCurrentIndex(mode)
        
        # Refresh current mode with file data
        if self._current_file_id:
            self._refresh_current_mode()
    
    def _refresh_current_mode(self):
        """Refresh current mode widget with file data."""
        mode = self._mode_combo.currentData()
        
        if mode == self.MODE_DETECTIONS and self._current_file_id:
            asyncio.ensure_future(
                self._detections_widget.load_detections(self._current_file_id)
            )
    
    @property
    def metadata_panel(self) -> MetadataPanel:
        return self._metadata
    
    def set_file(self, file_id: str):
        """Set current file to display by fetching record from database."""
        self._current_file_id = file_id
        
        if not file_id:
            self._metadata.clear()
            if self._detections_widget:
                self._detections_widget.clear()
            return
        
        # Load file for properties
        asyncio.ensure_future(self._load_file(file_id))
        
        # Refresh current mode
        self._refresh_current_mode()
    
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


class DetectionsWidget(QWidget):
    """
    Widget displaying object detection results for a file.
    
    Supports:
    - Legacy detections dict (FileRecord.detections)
    - New DetectionInstance records (relational)
    """
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        self._file_id = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Build detections UI."""
        from PySide6.QtWidgets import QListWidget, QScrollArea
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self._header = QLabel("Select a file to view detections")
        self._header.setStyleSheet("color: #888888; font-style: italic; padding: 8px;")
        layout.addWidget(self._header)
        
        # Detection list
        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d30;
                color: #e0e0e0;
                border: none;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3d3d40;
            }
            QListWidget::item:selected {
                background-color: #0d6efd;
            }
        """)
        layout.addWidget(self._list)
        
        # Stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #888888; font-size: 11px; padding: 4px;")
        layout.addWidget(self._stats_label)
    
    def clear(self):
        """Clear the widget."""
        self._file_id = None
        self._list.clear()
        self._header.setText("Select a file to view detections")
        self._stats_label.setText("")
    
    async def load_detections(self, file_id: str):
        """Load and display detections for a file."""
        from PySide6.QtWidgets import QListWidgetItem
        
        self._file_id = file_id
        self._list.clear()
        
        try:
            from src.ucorefs.models import FileRecord
            
            obj_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            record = await FileRecord.get(obj_id)
            
            if not record:
                self._header.setText("File not found")
                return
            
            self._header.setText(f"Detections for: {record.name}")
            
            total_detections = 0
            
            # Load NEW DetectionInstance records first
            try:
                from src.ucorefs.detection.models import DetectionInstance
                
                instances = await DetectionInstance.find({"file_id": obj_id})
                
                if instances:
                    # Group by class
                    by_class = {}
                    for inst in instances:
                        class_name = inst.class_name or "unknown"
                        if class_name not in by_class:
                            by_class[class_name] = []
                        by_class[class_name].append(inst)
                    
                    # Header
                    header_item = QListWidgetItem("🎯 DETECTION INSTANCES")
                    header_item.setForeground(Qt.GlobalColor.green)
                    self._list.addItem(header_item)
                    
                    for class_name, items in sorted(by_class.items()):
                        class_item = QListWidgetItem(f"   📦 {class_name} ({len(items)})")
                        class_item.setForeground(Qt.GlobalColor.cyan)
                        self._list.addItem(class_item)
                        
                        for inst in items[:5]:  # Show max 5 per class
                            conf = inst.confidence if hasattr(inst, 'confidence') else 0
                            bbox = inst.bbox if hasattr(inst, 'bbox') else None
                            bbox_str = f" @ {bbox}" if bbox else ""
                            det_item = QListWidgetItem(f"      • {conf:.1%}{bbox_str}")
                            self._list.addItem(det_item)
                            total_detections += 1
                        
                        if len(items) > 5:
                            more = QListWidgetItem(f"      ... +{len(items) - 5} more")
                            more.setForeground(Qt.GlobalColor.gray)
                            self._list.addItem(more)
                            total_detections += len(items) - 5
                    
                    self._list.addItem(QListWidgetItem(""))  # Spacer
                    
            except ImportError:
                pass  # DetectionInstance not available
            
            # Load LEGACY detections dict
            detections = record.detections or {}
            
            if detections:
                # Display each detection backend
                for backend, data in detections.items():
                    results = data.get("results", [])
                    model = data.get("model", "unknown")
                    
                    # Header for backend
                    header_item = QListWidgetItem(f"🔹 {backend.upper()} ({model})")
                    header_item.setForeground(Qt.GlobalColor.cyan)
                    self._list.addItem(header_item)
                    
                    if not results:
                        no_results = QListWidgetItem("   No objects detected")
                        no_results.setForeground(Qt.GlobalColor.gray)
                        self._list.addItem(no_results)
                        continue
                    
                    # Show each detection
                    for det in results[:10]:  # Limit to 10
                        label = det.get("label", det.get("class", "unknown"))
                        confidence = det.get("confidence", det.get("score", 0))
                        
                        item_text = f"   • {label}: {confidence:.1%}"
                        self._list.addItem(QListWidgetItem(item_text))
                        total_detections += 1
                    
                    if len(results) > 10:
                        more = QListWidgetItem(f"   ... +{len(results) - 10} more")
                        more.setForeground(Qt.GlobalColor.gray)
                        self._list.addItem(more)
                        total_detections += len(results) - 10
            
            # Show stats
            if total_detections > 0:
                self._stats_label.setText(f"Total: {total_detections} detections")
            else:
                item = QListWidgetItem("No detections found")
                item.setForeground(Qt.GlobalColor.gray)
                self._list.addItem(item)
                self._stats_label.setText("")
            
        except Exception as e:
            logger.error(f"Failed to load detections: {e}")
            self._header.setText(f"Error: {e}")

