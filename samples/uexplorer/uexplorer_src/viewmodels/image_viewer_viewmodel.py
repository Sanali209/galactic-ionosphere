"""
ImageViewerViewModel - Reactive state for the Image Viewer.
"""
from typing import Optional, List
from PySide6.QtCore import Signal
from loguru import logger
from bson import ObjectId

from src.ui.mvvm.viewmodel import BaseViewModel
from src.ui.mvvm.bindable import BindableProperty


class ImageViewerViewModel(BaseViewModel):
    """
    ViewModel for the Image Viewer.
    
    Acts as a bridge between global context (active_file, active_detection)
    and the image rendering/overlay logic.
    """
    
    # --- Synchronization Channels ---
    active_file_id = BindableProperty(sync_channel="active_file")
    active_detection_id = BindableProperty(sync_channel="active_detection")
    
    # --- View State (Local) ---
    zoom_level = BindableProperty(default=1.0)
    pan_x = BindableProperty(default=0.0)
    pan_y = BindableProperty(default=0.0)
    rotation = BindableProperty(default=0)
    
    # Interactive modes (e.g., browse, ROI select, pan)
    viewer_mode = BindableProperty(default="browse") # browse, roi, select
    
    # Signal emitted when image data needs reload (different from active_file_id changed)
    request_reload = Signal()

    def __init__(self, locator=None):
        super().__init__(locator)
        
        # When active_file changes, we reset zoom/pan
        self.active_file_idChanged.connect(self._on_active_file_changed)
        
        # Register for global sync
        self.initialize_reactivity()
        
        logger.info("ImageViewerViewModel initialized and registered for sync")

    def _on_active_file_changed(self, file_id):
        """Reset view when switching files."""
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.request_reload.emit()

    def zoom_in(self):
        self.zoom_level *= 1.2

    def zoom_out(self):
        self.zoom_level /= 1.2

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def select_detection(self, detection_id: str):
        """Set active detection (will broadcast to Properties Panel)."""
        self.active_detection_id = detection_id
