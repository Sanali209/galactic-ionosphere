"""
PropertiesViewModel - Reactive state for the Properties Panel.
"""
from typing import Optional, List
from PySide6.QtCore import Signal
from loguru import logger
from bson import ObjectId

from src.ui.mvvm.viewmodel import BaseViewModel
from src.ui.mvvm.bindable import BindableProperty, BindableList
from src.ucorefs.models.file_record import FileRecord


class PropertiesViewModel(BaseViewModel):
    """
    ViewModel for the Properties Panel.
    
    Acts as a bridge between the global context routing (active_file, active_detection)
    and the Properties Panel UI.
    """
    
    # --- Synchronization Channels ---
    # The file currently being inspected (synchronized with SelectionManager)
    active_file_id = BindableProperty(sync_channel="active_file")
    
    # The current selection (synchronized with SelectionManager)
    selected_file_ids = BindableProperty(sync_channel="selection")
    
    # The specific detection being focused (synchronized with Image Viewer)
    active_detection_id = BindableProperty(sync_channel="active_detection")
    
    # --- View State (Derived from active_file) ---
    file_name = BindableProperty(default="No file selected")
    file_path = BindableProperty(default="")
    file_size_str = BindableProperty(default="-")
    rating = BindableProperty(default=0)
    tags = BindableProperty(default=None)  # Will be initialized as BindableList
    description = BindableProperty(default="")
    
    # Details section (Reactive properties)
    dimensions_str = BindableProperty(default="-")
    created_at_str = BindableProperty(default="-")
    modified_at_str = BindableProperty(default="-")
    processing_status_text = BindableProperty(default="-")
    processing_status_color = BindableProperty(default="#888888")
    embeddings_summary = BindableProperty(default="-")
    detections_summary = BindableProperty(default="-")
    
    # Signal emitted when local state is refreshed from DB
    data_refreshed = Signal()
    
    # Signal emitted after debounce when a new file load should start (heavy operations)
    loading_requested = Signal(object) # file_id

    # --- Explicit Signals (Required for BindableProperty in PySide6) ---
    active_file_idChanged = Signal(object)
    selected_file_idsChanged = Signal(object)
    active_detection_idChanged = Signal(object)
    
    file_nameChanged = Signal(str)
    file_pathChanged = Signal(str)
    file_size_strChanged = Signal(str)
    ratingChanged = Signal(int)
    tagsChanged = Signal(object)
    descriptionChanged = Signal(str)
    
    dimensions_strChanged = Signal(str)
    created_at_strChanged = Signal(str)
    modified_at_strChanged = Signal(str)
    processing_status_textChanged = Signal(str)
    processing_status_colorChanged = Signal(str)
    embeddings_summaryChanged = Signal(str)
    detections_summaryChanged = Signal(str)

    def __init__(self, locator=None):
        super().__init__(locator)
        
        # Initialize BindableList for tags
        self.tags = BindableList()
        
        # Wire up internal reactive logic:
        # When the Global Context changes the active file ID, we fetch data.
        self.active_file_idChanged.connect(self._on_active_file_id_changed)
        
        # Register for synchronization via ContextSyncManager
        self.initialize_reactivity()
        
        logger.info("PropertiesViewModel initialized and registered for sync")

    def _on_active_file_id_changed(self, file_id):
        """Handle global active file change - trigger async load with debounce."""
        from PySide6.QtCore import QTimer
        
        # Initialize debounce timer if needed
        if not hasattr(self, '_load_timer'):
            self._load_timer = QTimer()
            self._load_timer.setSingleShot(True)
            self._load_timer.timeout.connect(self._do_debounced_load)
            self._pending_file_id = None
            
        self._pending_file_id = file_id
        self._load_timer.start(100) # 100ms debounce
        
    def _do_debounced_load(self):
        """Execute the actual load after debounce."""
        import asyncio
        file_id = self._pending_file_id
        
        # Notify sub-widgets to start their heavy loads (thumbnail, detections)
        self.loading_requested.emit(file_id)
        
        if file_id:
            asyncio.ensure_future(self.load_file_data(file_id))
        else:
            self.clear()

    async def load_file_data(self, file_id: str):
        """Fetch FileRecord from database and update view state."""
        try:
            # Handle both string and ObjectId
            obj_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            
            record = await FileRecord.get(obj_id)
            if record:
                self.file_name = record.name
                self.file_path = record.path
                self.rating = record.rating
                self.description = record.description or ""
                
                # Format size
                size = record.size or 0
                if size > 1024*1024:
                    self.file_size_str = f"{size / (1024*1024):.1f} MB"
                elif size > 1024:
                    self.file_size_str = f"{size / 1024:.1f} KB"
                else:
                    self.file_size_str = f"{size} B"
                
                # Update tags list (Reactive mutation)
                self.tags.clear()
                self.tags.extend(record.tags if record.tags else [])

                # Update details
                self.dimensions_str = f"{record.width or 0}x{record.height or 0}"
                self.created_at_str = record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "-"
                self.modified_at_str = record.modified_at.strftime("%Y-%m-%d %H:%M") if record.modified_at else "-"
                
                # Processing State logic
                from src.ucorefs.models.base import ProcessingState
                state = record.processing_state or 0
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
                self.processing_status_text = text
                self.processing_status_color = color
                
                # Embeddings summary
                embeddings = record.embeddings or {}
                available = [k.upper() for k in embeddings.keys() if k in ['clip', 'dino', 'blip']]
                self.embeddings_summary = ", ".join(available) if available else "None"
                
                # Detections summary (Async fetch)
                asyncio.ensure_future(self._update_detections_summary(obj_id))
                
                self.data_refreshed.emit()
                logger.debug(f"PropertiesViewModel: Loaded data for {record.name}")
            else:
                logger.warning(f"PropertiesViewModel: Record not found for {file_id}")
                self.clear()
                
        except Exception as e:
            logger.error(f"PropertiesViewModel: Failed to load file data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.clear()

    async def _update_detections_summary(self, file_id):
        """Fetch detections and build summary string."""
        try:
            from src.ucorefs.detection.service import DetectionService
            service = self.locator.get_system(DetectionService)
            detections = await service.get_detections(file_id)
            
            if not detections:
                self.detections_summary = "None"
                return
                
            labels = {}
            for det in detections:
                label = det.name.split('_')[0] if det.name else "unknown"
                labels[label] = labels.get(label, 0) + 1
            
            self.detections_summary = ", ".join(f"{c} {l}" for l, c in labels.items())
        except Exception as e:
            logger.debug(f"Failed to fetch detections summary: {e}")
            self.detections_summary = "Error"

    def clear(self):
        """Reset view state to defaults."""
        self.file_name = "No file selected"
        self.file_path = ""
        self.file_size_str = "-"
        self.rating = 0
        self.tags.clear()
        self.description = ""
        self.dimensions_str = "-"
        self.created_at_str = "-"
        self.modified_at_str = "-"
        self.processing_status_text = "-"
        self.processing_status_color = "#888888"
        self.embeddings_summary = "-"
        self.detections_summary = "-"
        self.data_refreshed.emit()

    def update_rating(self, rating: int):
        """Submit rating update to DB and broadcast via sync (if active_file.rating was synced)."""
        # For now, we update the record asynchronously
        import asyncio
        asyncio.ensure_future(self._save_rating(rating))
        
    async def _save_rating(self, rating: int):
        if not self.active_file_id: return
        try:
            record = await FileRecord.get(ObjectId(self.active_file_id))
            if record:
                record.rating = rating
                await record.save()
                self.rating = rating
                logger.info(f"Updated rating for {record.name} to {rating}")
        except Exception as e:
            logger.error(f"Failed to save rating: {e}")
