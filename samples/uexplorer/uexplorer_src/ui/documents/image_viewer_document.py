"""
ImageViewerDocument - Full-size image viewer and detection editor.

Opens when user double-clicks an image in the file browser.
Refactored to use QGraphicsView for bounding box editing.
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QToolBar, QButtonGroup, QToolButton
)
from src.ui.mvvm.data_context import BindableWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPixmap, QIcon, QPainter, QAction, QWheelEvent, QMouseEvent

from loguru import logger
from uexplorer_src.ui.widgets.detection_graphics_items import DetectionRectItem

class DetectionGraphicsView(BindableWidget, QGraphicsView):
    """
    Custom GraphicsView for image pan/zoom and detection editing.
    Reacts to ImageViewerViewModel.
    """
    
    # Modes
    MODE_VIEW = 0   # Pan/Zoom/Select
    MODE_DRAW = 1   # Draw Selection Box
    
    rect_created = Signal(QRectF)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = self.MODE_VIEW
        self._zoom_level = 1.0
        
        # Behavior
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        
        # Drawing state
        self._drawing_start: Optional[QPointF] = None
        self._current_draw_item: Optional[DetectionRectItem] = None
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
    
    def set_mode(self, mode: int):
        self._mode = mode
        if mode == self.MODE_VIEW:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode == self.MODE_DRAW:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            
    def wheelEvent(self, event: QWheelEvent):
        """Zoom on wheel."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom
            adj = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale(adj, adj)
            self._zoom_level *= adj
            event.accept()
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event: QMouseEvent):
        if self._mode == self.MODE_DRAW and event.button() == Qt.MouseButton.LeftButton:
            self._drawing_start = self.mapToScene(event.pos())
            # Create temp item
            self._current_draw_item = DetectionRectItem(
                self._drawing_start.x(), self._drawing_start.y(), 0, 0,
                label="New", color="#00ffff"
            )
            self.scene().addItem(self._current_draw_item)
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mode == self.MODE_DRAW and self._drawing_start and self._current_draw_item:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self._drawing_start, current_pos).normalized()
            self._current_draw_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mode == self.MODE_DRAW and self._drawing_start and self._current_draw_item:
            # Finish draw
            final_rect = self._current_draw_item.rect()
            
            # Helper: If too small, discard
            if final_rect.width() > 5 and final_rect.height() > 5:
                # Emit signal to Document to handle persistence/logic
                self.rect_created.emit(final_rect)
            
            # Remove temporary item (Controller will add proper model-backed item)
            self.scene().removeItem(self._current_draw_item)
            self._current_draw_item = None
            self._drawing_start = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _on_context_menu(self, pos):
        """Show context menu for detection items or empty area."""
        item = self.itemAt(pos)
        
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        
        if isinstance(item, DetectionRectItem):
            det_id = item.det_id
            
            act_edit = QAction("✏️ Edit Detection...", self)
            act_edit.triggered.connect(lambda: self.parent()._edit_detection(det_id))
            menu.addAction(act_edit)
            
            act_delete = QAction("🗑️ Delete", self)
            act_delete.triggered.connect(lambda: self.parent()._delete_detection(det_id))
            menu.addAction(act_delete)
            
            menu.addSeparator()
            
            act_similar = QAction("🔍 Find Similar in Area", self)
            act_similar.triggered.connect(lambda: self.parent()._find_similar_in_area(item.rect()))
            menu.addAction(act_similar)
            
        else:
            # Clicked on empty area or image
            act_roi = QAction("🎯 Find Similar in New ROI", self)
            act_roi.triggered.connect(self._enter_roi_mode)
            menu.addAction(act_roi)
            
        menu.exec(self.viewport().mapToGlobal(pos))

    def _enter_roi_mode(self):
        """Switch to ROI drawing mode."""
        from uexplorer_src.viewmodels.image_viewer_viewmodel import ImageViewerViewModel
        vm = self.get_typed_data_context(ImageViewerViewModel)
        if vm:
            vm.viewer_mode = "roi"


class ImageViewerDocument(BindableWidget):
    """
    Full-size image viewer document with detection editing support.
    Uses ImageViewerViewModel for state synchronization.
    """
    
    # Signals
    content_changed = Signal()
    
    def __init__(self, file_path: str, title: str = "Image", locator=None, parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self._title = title
        self._locator = locator
        self._file_record = None 
        
        # 1. Create ViewModel and set DataContext
        from uexplorer_src.viewmodels.image_viewer_viewmodel import ImageViewerViewModel
        self.viewmodel = ImageViewerViewModel(locator)
        self.set_data_context(self.viewmodel)

        # 2. Graphics components
        self._scene = QGraphicsScene()
        self._view = DetectionGraphicsView(parent=self)
        self._view.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        
        self._setup_ui()
        self._load_image()
        
        # 3. Connect reactive properties
        self.viewmodel.viewer_modeChanged.connect(self._on_viewer_mode_changed)
        self.viewmodel.request_reload.connect(self._load_image)
        self.viewmodel.active_detection_idChanged.connect(self._on_active_detection_changed)
        
        # Connect signals
        self._view.rect_created.connect(self._on_rect_draw_finished)
        
        # Load existing detections
        self._load_detections()
        
        logger.debug(f"ImageViewerDocument (Reactive) opened: {title}")

    def _on_viewer_mode_changed(self, mode: str):
        """Map ViewModel mode to GraphicsView behavior."""
        if mode == "draw" or mode == "roi":
            self._view.set_mode(DetectionGraphicsView.MODE_DRAW)
        else:
            self._view.set_mode(DetectionGraphicsView.MODE_VIEW)

    def _on_active_detection_changed(self, det_id: str):
        """Highlight detection when selected globally."""
        if not det_id:
            self._scene.clearSelection()
            return
            
        for item in self._scene.items():
            if isinstance(item, DetectionRectItem) and item.det_id == det_id:
                self._scene.clearSelection()
                item.setSelected(True)
                # Ensure it's visible in the view
                self._view.ensureVisible(item)
                logger.debug(f"ImageViewerDocument: Highlighted detection {det_id}")
                break
    
    def _setup_ui(self):
        """Build viewer UI and bind to ViewModel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setStyleSheet("background-color: #2d2d30; border-bottom: 1px solid #3d3d40;")
        
        # Actions
        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(self.fit_to_window)
        self._toolbar.addAction(act_fit)
        
        act_100 = QAction("1:1", self)
        act_100.triggered.connect(self.zoom_original)
        self._toolbar.addAction(act_100)
        
        self._toolbar.addSeparator()
        
        # Mode group (Reactive)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        
        btn_view = QToolButton()
        btn_view.setText("✋ View")
        btn_view.setCheckable(True)
        btn_view.setChecked(True)
        btn_view.clicked.connect(lambda: setattr(self.viewmodel, "viewer_mode", "browse"))
        self._toolbar.addWidget(btn_view)
        self._mode_group.addButton(btn_view)
        
        btn_draw = QToolButton()
        btn_draw.setText("✏️ Draw")
        btn_draw.setCheckable(True)
        btn_draw.clicked.connect(lambda: setattr(self.viewmodel, "viewer_mode", "draw"))
        self._toolbar.addWidget(btn_draw)
        self._mode_group.addButton(btn_draw)
        
        self._toolbar.addSeparator()
        
        # Toggle Detections
        self._chk_show_detections = QAction("Show Detections", self)
        self._chk_show_detections.setCheckable(True)
        self._chk_show_detections.setChecked(True)
        self._chk_show_detections.toggled.connect(self._toggle_detections)
        self._toolbar.addAction(self._chk_show_detections)
        
        self._toolbar.addSeparator()
        
        
        # Save Button
        self._btn_save = QToolButton()
        self._btn_save.setText("💾 Save Changes")
        self._btn_save.setToolTip("Save Detections (Ctrl+S)")
        self._btn_save.setShortcut("Ctrl+S")
        self._btn_save.clicked.connect(self.save_detections)
        self._btn_save.setEnabled(False) # Disabled initially
        
        # Style it to look important
        self._btn_save.setStyleSheet("""
            QToolButton { font-weight: bold; padding: 4px; }
            QToolButton:enabled { color: #4CAF50; border: 1px solid #4CAF50; border-radius: 4px; }
            QToolButton:disabled { color: #555; border: 1px solid #3d3d40; }
        """)
        
        self._toolbar.addWidget(self._btn_save)
        
        layout.addWidget(self._toolbar)
        layout.addWidget(self._view)
    
    def _load_image(self):
        """Load and display the image."""
        try:
            pixmap = QPixmap(self._file_path)
            
            if pixmap.isNull():
                self._scene.addText(f"Failed to load: {self._file_path}").setDefaultTextColor(Qt.GlobalColor.red)
                return
            
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            self._scene.addText(f"Error: {e}")
            
    def _load_detections(self):
        """Load detections from DB."""
        if not self._locator:
            return
            
        asyncio.create_task(self._fetch_detections())
            
    async def _fetch_detections(self):
        """Async fetch detections."""
        try:
            from src.ucorefs.models.file_record import FileRecord
            from src.ucorefs.detection.models import DetectionInstance
            
            # Find FileRecord first
            # We only have path? Ideally we should have ID.
            # But caller passed path. We need to resolve ID.
            # Or use Search to find file by path.
            
            record = await FileRecord.find_one({"path": self._file_path})
            if not record:
                 logger.warning(f"FileRecord not found for persistence: {self._file_path}")
                 return
                 
            self._file_record = record
            
            # Get image dimensions for coordinate conversion
            if not self._pixmap_item:
                logger.warning("No pixmap loaded, cannot convert bbox coordinates")
                return
            
            img_width = self._pixmap_item.pixmap().width()
            img_height = self._pixmap_item.pixmap().height()
            logger.debug(f"Image dimensions: {img_width}×{img_height}")
            
            # Fetch detections
            instances = await DetectionInstance.find({"file_id": record.id})
            logger.debug(f"Found {len(instances)} detection instances")
            
            # Add to scene with coordinate conversion
            for inst in instances:
                bbox = inst.bbox
                
                # Convert normalized (0-1) coordinates to pixel coordinates
                x_px = bbox.get('x', 0) * img_width
                y_px = bbox.get('y', 0) * img_height
                w_px = bbox.get('w', 0) * img_width
                h_px = bbox.get('h', 0) * img_height
                
                logger.debug(
                    f"Detection '{inst.name}': "
                    f"normalized=[{bbox.get('x'):.3f}, {bbox.get('y'):.3f}, {bbox.get('w'):.3f}×{bbox.get('h'):.3f}] → "
                    f"pixels=[{x_px:.1f}, {y_px:.1f}, {w_px:.1f}×{h_px:.1f}]"
                )
                
                self.add_detection(
                    x_px, y_px, w_px, h_px,
                    label=inst.name.split('_')[0] if '_' in inst.name else "Object",
                    score=inst.confidence,
                    editable=True,
                    det_id=str(inst.id)
                )
                
        except Exception as e:
            logger.error(f"Failed to fetch detections: {e}")

    def fit_to_window(self):
        """Fit image to view."""
        if self._pixmap_item:
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_original(self):
        """Zoom to 100%."""
        self._view.resetTransform()
        
    def _toggle_detections(self, checked: bool):
        """Show/Hide detection items."""
        for item in self._scene.items():
            if isinstance(item, DetectionRectItem):
                item.setVisible(checked)
                
    def _on_rect_draw_finished(self, rect: QRectF):
        """Handle new rect creation."""
        logger.info(f"New detection drawn: {rect}")
        
        from uexplorer_src.viewmodels.image_viewer_viewmodel import ImageViewerViewModel
        vm = self.get_typed_data_context(ImageViewerViewModel)
        if vm and vm.viewer_mode == "roi":
             # ROI search mode
             self._find_similar_in_area(rect)
             # Switch back to browse mode
             vm.viewer_mode = "browse"
             return

        # Create persistent item
        item = DetectionRectItem(rect.x(), rect.y(), rect.width(), rect.height(), label="New", score=1.0)
        item.set_editable(True)
        item.modified.connect(self._on_detection_modified)
            
        self._scene.addItem(item)
        
        # Select it
        self._scene.clearSelection()
        item.setSelected(True)
        
        # Signal content change
        self.content_changed.emit()
        self._btn_save.setEnabled(True) # Enable save button

    def showEvent(self, event):
        super().showEvent(event)
        # Fit on first show
        self.fit_to_window()

    # --- Detection Management ---
    
    def add_detection(self, x: float, y: float, w: float, h: float, label: str, score: float = 0.0, editable: bool = False, det_id: str = ""):
        """Add a detection box to the view."""
        item = DetectionRectItem(x, y, w, h, label=label, score=score, det_id=det_id)
        item.set_editable(editable)
        
        # Connect signals
        item.modified.connect(self._on_detection_modified)
        
        self._scene.addItem(item)
        
    def clear_detections(self):
        """Remove all detection items."""
        for item in list(self._scene.items()):
            if isinstance(item, DetectionRectItem):
                self._scene.removeItem(item)
                
    def _on_detection_modified(self):
        """Handle modification of existing detection."""
        self.content_changed.emit()
        self._btn_save.setEnabled(True)
        
    def save_detections(self):
        """Save current detections to DB."""
        if not self._locator or not self._file_record:
            logger.warning("Cannot save: No locator or FileRecord loaded")
            return
            
        detections = []
        for item in self._scene.items():
            if isinstance(item, DetectionRectItem):
                r = item.rect()
                detections.append({
                    "label": item._label,
                    "bbox": {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()},
                    "confidence": item._score
                })
        
        asyncio.create_task(self._save_async(detections))
        
    async def _save_async(self, detections: List[Dict]):
        """Execute save."""
        try:
            from src.ucorefs.detection.service import DetectionService
            
            service = self._locator.get_system(DetectionService)
            if not service:
                service = DetectionService(self._locator, self._locator.config) # Ad-hoc init if missing? Should be in locator.
                await service.initialize() # Warning: Side effect
                
            success = await service.update_file_detections(self._file_record.id, detections)
            
            if success:
                logger.info("Detections saved successfully")
                self._btn_save.setEnabled(False) # Disable after save
                self._btn_save.setText("✓ Saved")
                
                # Reset text after delay (could use QTimer, but this is async thread)
                # Just simplified visual feedback
                
            else:
                logger.error("Failed to save detections")
                
        except Exception as e:
            logger.error(f"Save failed: {e}")

    # --- Detection Handlers from Viewer ---

    def _edit_detection(self, det_id):
        """Open edit dialog for detection."""
        from src.ucorefs.detection.models import DetectionInstance
        from uexplorer_src.ui.dialogs.detection_edit_dialog import DetectionEditDialog
        from src.ucorefs.detection.service import DetectionService
        
        async def _do_edit():
            try:
                instance = await DetectionInstance.get(det_id)
                if not instance: return
                await instance.resolve_class_name()
                
                data = {
                    "name": instance.name,
                    "class_name": instance.class_name,
                    "group_name": instance.group_name,
                    "confidence": instance.confidence
                }
                
                dialog = DetectionEditDialog(self._locator, data, self)
                if dialog.exec():
                    result = dialog.get_result()
                    service = self._locator.get_system(DetectionService)
                    
                    if result['class_name'] != instance.class_name:
                         new_class = await service._get_or_create_class(result['class_name'])
                         result['detection_class_id'] = new_class.id
                         del result['class_name']
                    
                    if await service.update_instance(det_id, result):
                        logger.info(f"Updated detection {det_id}")
                        # Refresh locally
                        self.clear_detections()
                        await self._fetch_detections()
            except Exception as e:
                logger.error(f"Edit failed: {e}")

        asyncio.ensure_future(_do_edit())

    def _delete_detection(self, det_id):
        """Delete detection."""
        from src.ucorefs.detection.service import DetectionService
        
        async def _do_delete():
            service = self._locator.get_system(DetectionService)
            if await service.delete_instance(det_id):
                logger.info(f"Deleted detection {det_id}")
                # Remove from scene directly
                for item in self._scene.items():
                    if isinstance(item, DetectionRectItem) and item.det_id == det_id:
                        self._scene.removeItem(item)
                        break
        
        asyncio.ensure_future(_do_delete())

    def _find_similar_in_area(self, rect: QRectF):
        """Find images similar to the content of the given rect."""
        asyncio.ensure_future(self._perform_roi_search(rect))

    async def _perform_roi_search(self, rect: QRectF):
        """Execute ROI crop and similarity search."""
        try:
            if not self._pixmap_item: return
            
            # 1. Get Crop
            pixmap = self._pixmap_item.pixmap()
            
            # Clamp rect to pixmap bounds
            pix_rect = pixmap.rect()
            target_rect = rect.toRect().intersected(pix_rect)
            
            if target_rect.width() < 5 or target_rect.height() < 5:
                logger.warning("ROI too small for search")
                return
                
            crop = pixmap.copy(target_rect)
            
            # 2. Get Embedding via CLIPExtractor
            from src.ucorefs.extractors.clip_extractor import CLIPExtractor
            extractor = CLIPExtractor(self._locator)
            
            # Convert QPixmap to PIL
            from io import BytesIO
            from PIL import Image
            buffer = BytesIO()
            crop.save(buffer, "PNG")
            buffer.seek(0)
            pil_img = Image.open(buffer).convert("RGB")
            
            logger.info("Generating embedding for ROI crop...")
            embedding = await extractor.encode_image(pil_img)
            if not embedding:
                logger.warning("Failed to generate embedding for ROI")
                return
            
            # 3. Trigger Search via SearchPipeline
            from uexplorer_src.viewmodels.search_query import SearchQuery
            query = SearchQuery(
                mode="image",
                vector_query=embedding,
                limit=100
            )
            
            # Find SearchPipeline to execute
            main_window = self.window()
            if hasattr(main_window, 'search_pipeline'):
                 logger.info("Triggering ROI search on MainWindow pipeline")
                 await main_window.search_pipeline.execute(query)
            else:
                 logger.error("MainWindow.search_pipeline not found")
            
        except Exception as e:
            logger.error(f"ROI search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # --- Properties for docking/session ---
    
    @property
    def title(self) -> str:
        return self._title
    
    @property
    def file_path(self) -> str:
        return self._file_path
