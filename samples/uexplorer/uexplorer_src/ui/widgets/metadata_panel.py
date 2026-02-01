"""
Metadata Panel for UExplorer.
"""
from typing import TYPE_CHECKING, Optional
import asyncio
from datetime import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QScrollArea, QGroupBox, QFormLayout, QTextEdit,
                               QFrame, QSizePolicy, QToolButton)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from loguru import logger
from bson import ObjectId

from src.ui.mvvm.data_context import BindableWidget
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.services.fs_service import FSService
from src.ucorefs.thumbnails.service import ThumbnailService
try:
    from src.ui.widgets.tag_selector import TagSelector
except ImportError:
    try:
        from tag_selector import TagSelector
    except ImportError:
        # Fallback for relative import if loaded as package
        from .tag_selector import TagSelector

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator

class StarRating(QWidget):
    """Simple 5-star rating widget."""
    ratingChanged = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rating: int = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        
        self.stars = []
        for i in range(1, 6):
            btn = QToolButton()
            btn.setText("☆")  # Empty star
            btn.setStyleSheet("QToolButton { border: none; font-size: 18px; color: #ffa500; }")
            btn.clicked.connect(lambda _, r=i: self.set_rating(r, emit=True))
            btn.setFixedSize(24, 24)
            btn.setAutoRaise(True)
            layout.addWidget(btn)
            self.stars.append(btn)
            
    def set_rating(self, rating: int, emit: bool = False) -> None:
        self.rating = rating
        for i, btn in enumerate(self.stars):
            if i < rating:
                btn.setText("★")  # Filled star
            else:
                btn.setText("☆")  # Empty star
        
        if emit:
            self.ratingChanged.emit(rating)


class MetadataPanel(BindableWidget):
    """
    Panel for displaying and editing file metadata.
    """
    
    def __init__(self, locator: "ServiceLocator", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.locator: "ServiceLocator" = locator
        self.fs_service: FSService = locator.get_system(FSService)
        self.thumbnail_service: ThumbnailService = locator.get_system(ThumbnailService)
        
        self.current_file: Optional[FileRecord] = None
        
        self.init_ui()
    
    def set_data_context(self, vm, propagate=True):
        """Bind UI elements when ViewModel is connected."""
        super().set_data_context(vm, propagate)
        
        from uexplorer_src.viewmodels.properties_viewmodel import PropertiesViewModel
        if isinstance(vm, PropertiesViewModel):
            # Connect reactive properties
            vm.file_nameChanged.connect(self.name_label.setText)
            vm.file_size_strChanged.connect(self.size_label.setText)
            vm.ratingChanged.connect(lambda r: self.rating_widget.set_rating(r, emit=False))
            vm.descriptionChanged.connect(lambda d: self.desc_edit.setText(d))
            vm.active_file_idChanged.connect(self._on_active_file_id_changed)
            vm.loading_requested.connect(self._on_loading_requested)
            
            # Details section
            vm.dimensions_strChanged.connect(self.dim_label.setText)
            vm.created_at_strChanged.connect(self.created_label.setText)
            vm.modified_at_strChanged.connect(self.modified_label.setText)
            vm.processing_status_textChanged.connect(self.processing_label.setText)
            vm.processing_status_colorChanged.connect(
                lambda c: self.processing_label.setStyleSheet(f"color: {c}; font-weight: bold;")
            )
            vm.embeddings_summaryChanged.connect(self.embeddings_label.setText)
            vm.detections_summaryChanged.connect(self.detections_label.setText)
            
            logger.info("MetadataPanel connected to PropertiesViewModel")

    def _on_active_file_id_changed(self, file_id):
        """Handle file change to update UI enabled state."""
        if not file_id:
            self.clear()
            self.setEnabled(False)
        else:
            self.setEnabled(True)

    def _on_loading_requested(self, file_id):
        """Handle debounced loading request for heavy assets."""
        if file_id:
            import asyncio
            asyncio.ensure_future(self._load_thumb_by_id(file_id))

    async def _load_thumb_by_id(self, file_id):
        try:
            from bson import ObjectId
            obj_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            data = await self.thumbnail_service.get_or_create(obj_id, size=256)
            if data:
                pixmap = QPixmap()
                pixmap.loadFromData(data)
                scaled = pixmap.scaled(self.thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
            else:
                self.thumb_label.setText("No Preview")
        except Exception as e:
            logger.debug(f"MetadataPanel: Thumbnail load failed: {e}")
            self.thumb_label.setText("Error")
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setSpacing(15)
        
        # 1. Header (Thumbnail + Basic Info)
        self._init_header()
        
        # 2. Rating & Tags
        self._init_social()
        
        # 3. Description
        self._init_description()
        
        # 4. AI-Generated Caption (NEW)
        self._init_ai_section()
        
        # 5. XMP Metadata Buttons (NEW)
        self._init_xmp_buttons()
        
        # 6. EXIF / Details
        self._init_details()
        
        self.layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Start disabled
        self.setEnabled(False)
        
    def _init_header(self):
        group = QGroupBox("Preview")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QVBoxLayout(group)
        
        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setMinimumHeight(200)
        self.thumb_label.setStyleSheet("background-color: #1e1e1e; border-radius: 4px; color: #888888;")
        layout.addWidget(self.thumb_label)
        
        # Info
        self.name_label = QLabel("Filename.jpg")
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.name_label)
        
        info_layout = QHBoxLayout()
        self.size_label = QLabel("0 KB")
        self.type_label = QLabel("JPG Image")
        info_layout.addWidget(self.size_label)
        info_layout.addStretch()
        info_layout.addWidget(self.type_label)
        layout.addLayout(info_layout)
        
        self.layout.addWidget(group)
        
    def _init_social(self):
        group = QGroupBox("Organization")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QFormLayout(group)
        
        self.rating_widget = StarRating()
        self.rating_widget.ratingChanged.connect(self._save_rating)
        layout.addRow("Rating:", self.rating_widget)
        
        self.tag_selector = TagSelector(self.locator)
        self.tag_selector.tags_changed.connect(self._save_tags)
        layout.addRow("Tags:", self.tag_selector)
        
        self.layout.addWidget(group)
        
    def _init_description(self):
        group = QGroupBox("Description")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QVBoxLayout(group)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        self.desc_edit.setPlaceholderText("Add a description...")
        # Save on focus lost? hard in Qt without event filter.
        # Add a save button?
        # Or rely on explicit "Save" action?
        # For auto-save, we can use debounced textChanged
        self.desc_edit.textChanged.connect(self._on_desc_changed)
        
        layout.addWidget(self.desc_edit)
        self.layout.addWidget(group)
    
    def _init_ai_section(self):
        """AI-generated caption display section."""
        group = QGroupBox("AI-Generated Content")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QVBoxLayout(group)
        
        self.ai_caption_label = QLabel("(No AI description yet)")
        self.ai_caption_label.setWordWrap(True)
        self.ai_caption_label.setStyleSheet("color: #9a8a5a; font-style: italic; padding: 5px;")
        self.ai_caption_label.setMinimumHeight(40)
        layout.addWidget(self.ai_caption_label)
        
        # Button to generate/regenerate description
        from PySide6.QtWidgets import QPushButton
        btn_generate = QPushButton("Generate Description (BLIP)")
        btn_generate.setStyleSheet("QPushButton { background-color: #3d3d3d; padding: 5px; }")
        btn_generate.clicked.connect(self._trigger_blip)
        layout.addWidget(btn_generate)
        
        self.layout.addWidget(group)
    
    def _init_xmp_buttons(self):
        """XMP metadata read/write buttons."""
        group = QGroupBox("XMP Metadata")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QHBoxLayout(group)
        
        from PySide6.QtWidgets import QPushButton
        
        btn_read_xmp = QPushButton("Read from XMP")
        btn_read_xmp.setStyleSheet("QPushButton { background-color: #3d3d3d; padding: 5px; }")
        btn_read_xmp.clicked.connect(self._read_xmp_metadata)
        layout.addWidget(btn_read_xmp)
        
        btn_write_xmp = QPushButton("Write to XMP")
        btn_write_xmp.setStyleSheet("QPushButton { background-color: #3d3d3d; padding: 5px; }")
        btn_write_xmp.clicked.connect(self._write_xmp_metadata)
        layout.addWidget(btn_write_xmp)
        
        self.layout.addWidget(group)
        
    def _init_details(self):
        self.details_group = QGroupBox("Details")
        self.details_group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.details_layout = QFormLayout(self.details_group)
        
        self.dim_label = QLabel("-")
        self.created_label = QLabel("-")
        self.modified_label = QLabel("-")
        
        # ProcessingState indicator - demonstrates UCoreFS pipeline status
        self.processing_label = QLabel("-")
        self.processing_label.setStyleSheet("color: #888888;")
        
        # Embeddings status - shows available embeddings (CLIP, DINO)
        self.embeddings_label = QLabel("-")
        self.embeddings_label.setStyleSheet("color: #888888;")
        
        # Detection count - shows detected objects/faces
        self.detections_label = QLabel("-")
        self.detections_label.setStyleSheet("color: #888888;")
        
        self.details_layout.addRow("Dimensions:", self.dim_label)
        self.details_layout.addRow("Created:", self.created_label)
        self.details_layout.addRow("Modified:", self.modified_label)
        self.details_layout.addRow("Processing:", self.processing_label)
        self.details_layout.addRow("Embeddings:", self.embeddings_label)
        self.details_layout.addRow("Detections:", self.detections_label)
        
        self.layout.addWidget(self.details_group)

        # Update details
        # ...
        
    def clear(self):
        """Reset panel."""
        self.current_file = None
        self.setEnabled(False)
        self.name_label.setText("-")
        self.thumb_label.clear()
        self.thumb_label.setText("No Selection")
        self.rating_widget.set_rating(0)
        self.desc_edit.clear()
        
    async def _load_thumb(self, record):
        try:
            data = await self.thumbnail_service.get_or_create(record._id, size=256)
            if data:
                pixmap = QPixmap()
                pixmap.loadFromData(data)
                # Scale to fit
                scaled = pixmap.scaled(self.thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
            else:
                self.thumb_label.setText("No Preview")
        except Exception:
            self.thumb_label.setText("Error")

    def _save_rating(self, rating):
        if self.current_file:
            # Optimistic update
            self.current_file.rating = rating
            asyncio.ensure_future(self._persist_change())

    def _save_tags(self, tag_ids):
        if self.current_file:
            self.current_file.tag_ids = tag_ids
            asyncio.ensure_future(self._persist_change())
            
            # Refresh tag tree in main window if accessible
            try:
                from PySide6.QtWidgets import QApplication
                main_window = QApplication.instance().main_window
                if hasattr(main_window, 'tags_tree'):
                    asyncio.ensure_future(main_window.tags_tree.refresh_tags())
            except:
                pass
            
    def _on_desc_changed(self):
        # Debounce? For now just save on every char is too much db write.
        # But we don't have a timer here. 
        # Ideally, wait for focus out.
        if self.current_file:
            self.current_file.description = self.desc_edit.toPlainText()
            # Don't persist immediately on every keystroke
            # Maybe start a timer?
            pass

    async def _persist_change(self):
        if self.current_file:
            try:
                await self.current_file.save()
                logger.info(f"Saved metadata for {self.current_file.name}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    # Explicit save for description
    def save_description(self):
         asyncio.ensure_future(self._persist_change())

    def _update_processing_state(self, record: FileRecord):
        """
        Display ProcessingState with color-coded indicator.
        
        Demonstrates UCoreFS pipeline phases:
        - DISCOVERED (0) -> REGISTERED (10) -> METADATA_READY (20)
        - THUMBNAIL_READY (30) -> INDEXED (40) -> ANALYZED (50) -> COMPLETE (100)
        """
        from src.ucorefs.models.base import ProcessingState
        
        state = getattr(record, 'processing_state', 0)
        
        # Map state to display text and color
        state_map = {
            ProcessingState.DISCOVERED: ("Discovered", "#888888"),
            ProcessingState.REGISTERED: ("Registered", "#6a8aba"),
            ProcessingState.METADATA_READY: ("Metadata Ready", "#5a9aca"),
            ProcessingState.THUMBNAIL_READY: ("Thumbnailed", "#5aaa8a"),
            ProcessingState.INDEXED: ("Indexed", "#7aaa5a"),
            ProcessingState.ANALYZED: ("AI Analyzed", "#9a8a5a"),
            ProcessingState.COMPLETE: ("✓ Complete", "#5aca5a"),
        }
        
        text, color = state_map.get(state, (f"State {state}", "#888888"))
        self.processing_label.setText(text)
        self.processing_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_embeddings_status(self, record: FileRecord):
        """
        Display available embeddings for the file.
        
        Shows which embedding models have processed this file (CLIP, DINO).
        """
        embeddings = getattr(record, 'embeddings', None) or {}
        
        if not embeddings:
            self.embeddings_label.setText("None")
            self.embeddings_label.setStyleSheet("color: #888888;")
            return
        
        # Build status text
        available = []
        if 'clip' in embeddings:
            available.append("CLIP")
        if 'dino' in embeddings:
            available.append("DINO")
        if 'blip' in embeddings:
            available.append("BLIP")
        
        if available:
            self.embeddings_label.setText(", ".join(available))
            self.embeddings_label.setStyleSheet("color: #5aca5a; font-weight: bold;")
        else:
            # Has embeddings dict but no recognized keys
            self.embeddings_label.setText(f"{len(embeddings)} custom")
            self.embeddings_label.setStyleSheet("color: #9a8a5a;")
    
    async def _update_detections_count(self, record: FileRecord):
        """
        Display count of detected objects for the file.
        
        Queries DetectionService for DetectionInstance records.
        """
        try:
            from src.ucorefs.detection import DetectionService
            
            detection_service = self.locator.get_system(DetectionService)
            detections = await detection_service.get_detections(record._id)
            
            count = len(detections)
            if count > 0:
                # Group by label
                labels = {}
                for det in detections:
                    label = getattr(det, 'name', 'unknown').split('_')[0]
                    labels[label] = labels.get(label, 0) + 1
                
                summary = ", ".join(f"{c} {l}" for l, c in labels.items())
                self.detections_label.setText(summary)
                self.detections_label.setStyleSheet("color: #5aca5a; font-weight: bold;")
            else:
                self.detections_label.setText("None")
                self.detections_label.setStyleSheet("color: #888888;")
                
        except (KeyError, ImportError):
            # DetectionService not available
            self.detections_label.setText("-")
            self.detections_label.setStyleSheet("color: #888888;")
        except Exception as e:
            logger.debug(f"Failed to get detections: {e}")
            self.detections_label.setText("-")
    
    def _trigger_blip(self):
        """Manually trigger BLIP caption generation for current file via EngineProxy."""
        if not self.current_file:
            return
        
        try:
            from src.core.engine.proxy import EngineProxy
            engine_proxy = self.locator.get_system(EngineProxy)
            
            if not engine_proxy:
                logger.error("EngineProxy not available")
                return
            
            file_id = self.current_file._id
            
            async def _enqueue_phase3():
                from src.core.locator import get_active_locator
                from src.ucorefs.processing.pipeline import ProcessingPipeline
                sl = get_active_locator()
                pipeline = sl.get_system(ProcessingPipeline)
                await pipeline.enqueue_phase3(file_id)
            
            engine_proxy.submit(_enqueue_phase3())
            logger.info(f"Queued Phase 3 (BLIP) processing for {self.current_file.name}")
        except Exception as e:
            logger.error(f"Failed to queue BLIP processing: {e}")
    
    async def _read_xmp_metadata(self):
        """Re-extract XMP metadata from file and force update FileRecord."""
        if not self.current_file:
            return
        
        try:
            from src.ucorefs.extractors.xmp import xmp_extractor
            from src.ucorefs.extractors.metadata import MetadataExtractor
            
            if not xmp_extractor.is_available():
                logger.warning("pyexiv2 not available")
                return
            
            # Force re-extract XMP
            xmp_data = xmp_extractor.extract(self.current_file.path)
            
            if xmp_data:
                # Force update even if fields not empty
                if xmp_data.get("label"):
                    self.current_file.label = xmp_data["label"]
                
                if xmp_data.get("description"):
                    self.current_file.description = xmp_data["description"]
                
                # Extract rating
                raw_xmp = xmp_data.get("raw_xmp", {})
                if "Xmp.xmp.Rating" in raw_xmp:
                    try:
                        self.current_file.rating = int(raw_xmp["Xmp.xmp.Rating"])
                    except (ValueError, TypeError):
                        pass
                
                # Update tags
                if xmp_data.get("tags"):
                    extractor = MetadataExtractor(self.locator)
                    resolved_ids = await extractor._resolve_tags(xmp_data["tags"])
                    existing = set(self.current_file.tag_ids)
                    existing.update(resolved_ids)
                    self.current_file.tag_ids = list(existing)
                
                await self.current_file.save()
                logger.info(f"Re-imported XMP metadata for {self.current_file.name}")
                
                # Refresh display
                self.set_file(self.current_file)
            else:
                logger.info(f"No XMP data found in {self.current_file.path}")
                
        except Exception as e:
            logger.error(f"Failed to read XMP metadata: {e}")
    
    async def _write_xmp_metadata(self):
        """Write FileRecord metadata back to XMP sidecar file."""
        if not self.current_file:
            return
        
        try:
            import pyexiv2
            from pathlib import Path
            
            # Create XMP sidecar path (filename.ext.xmp)
            sidecar_path = str(Path(self.current_file.path) + Path('.xmp'))
            
            # Prepare XMP data
            xmp_dict = {}
            
            if self.current_file.label:
                xmp_dict["Xmp.xmp.Label"] = self.current_file.label
            
            if self.current_file.description:
                xmp_dict["Xmp.dc.description"] = {"x-default": self.current_file.description}
            
            if self.current_file.rating > 0:
                xmp_dict["Xmp.xmp.Rating"] = str(self.current_file.rating)
            
            # TODO: Export tags to Xmp.dc.subject
            # Would need to get tag names from TagManager
            
            # Write XMP sidecar
            img = pyexiv2.ImageData()
            img.modify_xmp(xmp_dict)
            with open(sidecar_path, 'wb') as f:
                f.write(img.get_bytes())
            img.close()
            
            logger.info(f"Wrote XMP sidecar to {sidecar_path}")
            
        except ImportError:
            logger.warning("pyexiv2 not available for XMP write")
        except Exception as e:
            logger.error(f"Failed to write XMP metadata: {e}")

