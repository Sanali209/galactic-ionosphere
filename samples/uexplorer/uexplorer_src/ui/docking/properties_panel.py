"""
Dockable Properties Panel (Metadata) for UExplorer.

Supports multiple modes: Properties, Detections.
Works with DockingService (QWidget-based).
"""
from typing import TYPE_CHECKING, Optional
import asyncio
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QStackedWidget, QWidget, QMenu
)
from PySide6.QtGui import QAction, QColor
from PySide6.QtCore import Qt, Signal, Slot
from bson import ObjectId
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.metadata_panel import MetadataPanel

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


from src.ui.mvvm.bindable import BindableProperty
from src.ui.mvvm.data_context import BindableWidget
from PySide6.QtCore import Signal


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
        
        super().__init__(locator, parent)
        
        # Initialization
        
        # Create and set ViewModel for this panel hierarchy
        from uexplorer_src.viewmodels.properties_viewmodel import PropertiesViewModel
        vm = PropertiesViewModel(locator)
        self.set_data_context(vm)
        
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
        
        # Explicitly load detections when switching to Detections mode
        vm = self.get_typed_data_context(PropertiesViewModel)
        file_id = vm.active_file_id if vm else None
        
        if mode == self.MODE_DETECTIONS and file_id:
            logger.debug(f"Switching to Detections mode for file {file_id}")
            asyncio.ensure_future(
                self._detections_widget.load_detections(file_id)
            )
    
    def _refresh_current_mode(self):
        """Refresh current mode widget with file data."""
        mode = self._mode_combo.currentData()
        
        vm = self.get_typed_data_context(PropertiesViewModel)
        file_id = vm.active_file_id if vm else None
        
        if mode == self.MODE_DETECTIONS and file_id:
            logger.info(f"[PropertiesPanel] Triggering detection load for file_id={file_id}")
            asyncio.ensure_future(
                self._detections_widget.load_detections(file_id)
            )
        else:
            logger.debug(f"[PropertiesPanel] Not loading detections: mode={mode}, has_file={bool(file_id)}")
    
    @property
    def metadata_panel(self) -> MetadataPanel:
        return self._metadata
    
    # set_file is now redundant as we use DataContext synchronization.
    # The panel reacts via MetadataPanel and DetectionsWidget which 
    # both observe the PropertiesViewModel in their DataContext.


class DetectionsWidget(BindableWidget):
    """
    Widget displaying object detection results for a file.
    
    Refactored to use QTreeWidget for hierarchical display.
    """
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        self._file_id = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Build detections UI with hierarchical tree."""
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self._header = QLabel("Select a file to view detections")
        self._header.setStyleSheet("color: #888888; font-style: italic; padding: 8px;")
        layout.addWidget(self._header)
        
        # Detection tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(15)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2d2d30;
                color: #e0e0e0;
                border: none;
            }
            QTreeWidget::item {
                padding: 4px;
                border-bottom: 1px solid #3d3d40;
            }
            QTreeWidget::item:selected {
                background-color: #0d6efd;
            }
        """)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree)
        
        # Stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #888888; font-size: 11px; padding: 4px;")
        layout.addWidget(self._stats_label)

    def set_data_context(self, vm, propagate=True):
        """Connect to ViewModel properties."""
        super().set_data_context(vm, propagate)
        
        from uexplorer_src.viewmodels.properties_viewmodel import PropertiesViewModel
        if isinstance(vm, PropertiesViewModel):
            vm.active_file_idChanged.connect(self._on_active_file_id_changed)
            vm.loading_requested.connect(self._on_loading_requested)
            logger.info("DetectionsWidget connected to PropertiesViewModel")

    def _on_active_file_id_changed(self, file_id):
        """Update UI immediately when active file changes."""
        if not file_id:
            self.clear()

    def _on_loading_requested(self, file_id):
        """Handle debounced loading request for heavy detections."""
        if file_id:
            asyncio.ensure_future(self.load_detections(file_id))

    def _on_tree_selection_changed(self):
        """Handle tree selection and publish to active_detection channel."""
        items = self._tree.selectedItems()
        if not items:
            return
            
        item = items[0]
        det_id = item.data(0, Qt.UserRole)
        
        if det_id:
            vm = self.get_typed_data_context(PropertiesViewModel)
            if vm:
                vm.active_detection_id = det_id
                logger.debug(f"DetectionsWidget: Selected detection {det_id}")
    
    def clear(self):
        """Clear the widget."""
        self._file_id = None
        self._tree.clear()
        self._header.setText("Select a file to view detections")
        self._stats_label.setText("")

    def _on_context_menu(self, pos):
        """Show context menu for detection items."""
        item = self._tree.itemAt(pos)
        if not item:
            return
            
        det_id = item.data(0, Qt.UserRole)
        if not det_id:
            return
            
        menu = QMenu(self)
        
        act_edit = QAction("✏️ Edit Detection...", self)
        act_edit.triggered.connect(lambda: self._edit_detection(det_id))
        menu.addAction(act_edit)
        
        act_delete = QAction("🗑️ Delete", self)
        act_delete.triggered.connect(lambda: self._delete_detection(det_id))
        menu.addAction(act_delete)
        
        menu.exec(self._tree.mapToGlobal(pos))

    def _edit_detection(self, det_id):
        """Open edit dialog and update detection."""
        from src.ucorefs.detection.models import DetectionInstance
        from uexplorer_src.ui.dialogs.detection_edit_dialog import DetectionEditDialog
        from src.ucorefs.detection.service import DetectionService
        
        async def _do_edit():
            try:
                # Fetch full instance data
                instance = await DetectionInstance.get(det_id)
                if not instance:
                    return
                
                await instance.resolve_class_name()
                
                data = {
                    "name": instance.name,
                    "class_name": instance.class_name,
                    "group_name": instance.group_name,
                    "confidence": instance.confidence
                }
                
                # Show dialog (synchronous exec)
                dialog = DetectionEditDialog(self.locator, data, self)
                if dialog.exec():
                    result = dialog.get_result()
                    
                    # Update via service
                    service = self.locator.get_system(DetectionService)
                    
                    # Resolve class ID if it changed
                    if result['class_name'] != instance.class_name:
                         new_class = await service._get_or_create_class(result['class_name'])
                         result['detection_class_id'] = new_class.id
                         del result['class_name']
                    
                    success = await service.update_instance(det_id, result)
                    if success:
                        logger.info(f"Updated detection {det_id}")
                        # Reload tree
                        if self._file_id:
                            await self.load_detections(self._file_id)
            except Exception as e:
                logger.error(f"Edit failed: {e}")

        asyncio.ensure_future(_do_edit())

    def _delete_detection(self, det_id):
        """Delete detection via service."""
        from src.ucorefs.detection.service import DetectionService
        
        async def _do_delete():
            service = self.locator.get_system(DetectionService)
            if await service.delete_instance(det_id):
                logger.info(f"Deleted detection {det_id}")
                if self._file_id:
                    await self.load_detections(self._file_id)
        
        asyncio.ensure_future(_do_delete())

    async def load_detections(self, file_id: str):
        """Load and display detections for a file hierarchically."""
        from PySide6.QtWidgets import QTreeWidgetItem
        from PySide6.QtGui import QColor
        
        logger.debug(f"[DetectionsWidget] Loading detections for file_id={file_id}")
        
        self._file_id = file_id
        self._tree.clear()
        
        try:
            from src.ucorefs.models import FileRecord
            
            obj_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            record = await FileRecord.get(obj_id)
            
            if not record:
                self._header.setText("File not found")
                return
            
            self._header.setText(f"Detections for: {record.name}")
            
            total_detections = 0
            
            # 1. Load NEW DetectionInstance records (Relational)
            try:
                from src.ucorefs.detection.models import DetectionInstance
                
                pipeline = [
                    {"$match": {"file_id": obj_id}},
                    {"$lookup": {
                        "from": "detection_classes",
                        "localField": "detection_class_id",
                        "foreignField": "_id",
                        "as": "detection_class"
                    }},
                    {"$unwind": {
                        "path": "$detection_class",
                        "preserveNullAndEmptyArrays": True
                    }},
                    {"$project": {
                        "class_name": "$detection_class.class_name",
                        "group_name": 1,
                        "bbox": 1,
                        "confidence": 1,
                        "name": 1
                    }}
                ]
                
                collection = DetectionInstance.get_collection()
                results = await collection.aggregate(pipeline).to_list(length=None)
                
                if results:
                    # Group by class → group_name
                    by_class = {}
                    for doc in results:
                        class_name = doc.get('class_name') or "unknown"
                        group_name = doc.get('group_name') or "unknown"
                        
                        if class_name not in by_class:
                            by_class[class_name] = {}
                        if group_name not in by_class[class_name]:
                            by_class[class_name][group_name] = []
                        
                        by_class[class_name][group_name].append(doc)
                    
                    # Add to tree
                    for class_name, groups in sorted(by_class.items()):
                        total_count = sum(len(items) for items in groups.values())
                        class_node = QTreeWidgetItem(self._tree, [f"📦 {class_name} ({total_count})"])
                        class_node.setForeground(0, QColor("#0dcaf0")) # Cyan
                        
                        for group_name, items in sorted(groups.items()):
                            group_node = QTreeWidgetItem(class_node, [f"├─ {group_name} ({len(items)})"])
                            group_node.setForeground(0, QColor("#ffc107")) # Yellow
                            
                            for doc in items:
                                conf = doc.get('confidence', 0)
                                bbox = doc.get('bbox', {})
                                det_id = str(doc.get('_id'))
                                
                                if bbox and all(k in bbox for k in ['x', 'y', 'w', 'h']):
                                    bbox_str = f"[{bbox['x']:.2f}, {bbox['y']:.2f}, {bbox['w']:.2f}×{bbox['h']:.2f}]"
                                else:
                                    bbox_str = "no bbox"
                                
                                det_node = QTreeWidgetItem(group_node, [f"• {conf:.1%} @ {bbox_str}"])
                                det_node.setData(0, Qt.UserRole, det_id)
                                total_detections += 1
                        
                        class_node.setExpanded(True)
                
            except Exception as e:
                logger.warning(f"Failed to load DetectionInstances: {e}")
            
            # 2. Load LEGACY detections dict
            detections = record.detections or {}
            if detections:
                for backend, data in detections.items():
                    results = data.get("results", [])
                    model = data.get("model", "unknown")
                    
                    legacy_root = QTreeWidgetItem(self._tree, [f"🔹 {backend.upper()} (Legacy)"])
                    legacy_root.setForeground(0, QColor("#6c757d")) # Gray
                    
                    for det in results:
                        label = det.get("label", det.get("class", "unknown"))
                        conf = det.get("confidence", det.get("score", 0))
                        
                        item = QTreeWidgetItem(legacy_root, [f"• {label}: {conf:.1%}"])
                        total_detections += 1
            
            # Show stats
            if total_detections > 0:
                self._stats_label.setText(f"Total: {total_detections} detections")
            else:
                QTreeWidgetItem(self._tree, ["No detections found"])
                self._stats_label.setText("")
            
        except Exception as e:
            logger.error(f"Failed to load detections: {e}")
            self._header.setText(f"Error: {e}")

