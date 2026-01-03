"""
Dockable Similar Items Panel for UExplorer.

Shows files similar to the current selection using vector similarity.
Works with DockingService (QWidget-based).
"""
from typing import TYPE_CHECKING, List, Optional
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QSlider, QSpinBox, QWidget
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap
import asyncio
from pathlib import Path
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator
    from src.ucorefs.ai.similarity_service import SimilarityService
    from src.ucorefs.thumbnails.service import ThumbnailService


class SimilarItemsPanel(PanelBase):
    """
    Dockable panel showing files similar to current selection.
    
    Connects to SelectionManager and uses SimilarityService
    for vector-based similarity search.
    """
    
    # Emitted when similar item is clicked (for navigation)
    item_clicked = Signal(str)  # file_id
    
    def __init__(self, parent: Optional[QWidget], locator: "ServiceLocator") -> None:
        self._list: Optional[QListWidget] = None
        self._similarity_service: Optional["SimilarityService"] = None
        self._thumbnail_service: Optional["ThumbnailService"] = None
        self._selection_manager: Optional[object] = None
        self._current_file_id: Optional[str] = None
        self._threshold_slider: Optional[QSlider] = None
        self._limit_spinbox: Optional[QSpinBox] = None
        super().__init__(locator, parent)
        
        # Get services
        try:
            from src.ucorefs.ai.similarity_service import SimilarityService
            self._similarity_service = locator.get_system(SimilarityService)
        except (KeyError, ImportError):
            logger.warning("SimilarityService not available")
        
        try:
            from src.ucorefs.thumbnails.service import ThumbnailService
            self._thumbnail_service = locator.get_system(ThumbnailService)
        except (KeyError, ImportError):
            logger.warning("ThumbnailService not available")
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("🔍 Similar Files")
        title.setStyleSheet("font-weight: bold; color: #ffffff;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Refresh button
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedSize(24, 24)
        self.btn_refresh.setToolTip("Refresh Similar Items")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #5a7aaa;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #6a8aba; }
        """)
        self.btn_refresh.clicked.connect(self._on_refresh)
        header.addWidget(self.btn_refresh)
        
        layout.addLayout(header)
        
        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(50, 99)  # 0.50 to 0.99
        self._threshold_slider.setValue(85)  # 0.85 default
        self._threshold_slider.setToolTip("Minimum similarity score")
        threshold_layout.addWidget(self._threshold_slider)
        
        self._threshold_label = QLabel("0.85")
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_label.setText(f"{v/100:.2f}")
        )
        threshold_layout.addWidget(self._threshold_label)
        
        layout.addLayout(threshold_layout)
        
        # Results list
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
            QListWidget::item:hover {
                background-color: #3d3d40;
            }
        """)
        self._list.setIconSize(self._list.size() / 4)  # Small icons
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)
        
        # Status label
        self._status_label = QLabel("Select a file to find similar items")
        self._status_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self._status_label)
        
        # Duplicate marking buttons
        dup_layout = QHBoxLayout()
        dup_layout.setSpacing(4)
        
        dup_label = QLabel("Mark as:")
        dup_label.setStyleSheet("color: #aaaaaa;")
        dup_layout.addWidget(dup_label)
        
        self.btn_exact_dup = QPushButton("Exact")
        self.btn_exact_dup.setToolTip("Mark as Exact Duplicate")
        self.btn_exact_dup.clicked.connect(lambda: self._mark_duplicate("exact_duplicate"))
        self.btn_exact_dup.setStyleSheet(self._dup_button_style("#c0392b"))
        dup_layout.addWidget(self.btn_exact_dup)
        
        self.btn_near_dup = QPushButton("Near")
        self.btn_near_dup.setToolTip("Mark as Near Duplicate")
        self.btn_near_dup.clicked.connect(lambda: self._mark_duplicate("near_duplicate"))
        self.btn_near_dup.setStyleSheet(self._dup_button_style("#e67e22"))
        dup_layout.addWidget(self.btn_near_dup)
        
        self.btn_similar = QPushButton("Similar")
        self.btn_similar.setToolTip("Mark as Similar (Keep Both)")
        self.btn_similar.clicked.connect(lambda: self._mark_duplicate("similar"))
        self.btn_similar.setStyleSheet(self._dup_button_style("#27ae60"))
        dup_layout.addWidget(self.btn_similar)
        
        self.btn_same_set = QPushButton("Set")
        self.btn_same_set.setToolTip("Mark as Same Set")
        self.btn_same_set.clicked.connect(lambda: self._mark_duplicate("same_set"))
        self.btn_same_set.setStyleSheet(self._dup_button_style("#2980b9"))
        dup_layout.addWidget(self.btn_same_set)
        
        dup_layout.addStretch()
        layout.addLayout(dup_layout)
    
    def _dup_button_style(self, color: str) -> str:
        """Get duplicate button style."""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {color}cc; }}
            QPushButton:disabled {{ background-color: #555555; }}
        """
    
    def _mark_duplicate(self, duplicate_type: str):
        """Mark selected similar item as duplicate."""
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QTimer
        
        selected_items = self._list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "Mark Duplicate", 
                "Select a similar item from the list to mark as duplicate."
            )
            return
        
        target_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        if not self._current_file_id or not target_id:
            return
        
        async def do_mark():
            try:
                from bson import ObjectId
                from src.ucorefs.ai.similarity_service import SimilarityService
                
                if not self._similarity_service:
                    QTimer.singleShot(0, lambda: QMessageBox.warning(
                        self, "Error", "SimilarityService not available."
                    ))
                    return
                
                result = await self._similarity_service.mark_as_duplicate(
                    ObjectId(self._current_file_id),
                    ObjectId(target_id),
                    duplicate_type
                )
                
                if result.get("success"):
                    QTimer.singleShot(0, lambda: QMessageBox.information(
                        self, "Marked", f"Marked as {duplicate_type.replace('_', ' ')}"
                    ))
                else:
                    QTimer.singleShot(0, lambda: QMessageBox.warning(
                        self, "Error", result.get("error", "Failed to mark")
                    ))
                    
            except Exception as e:
                logger.exception(f"Mark duplicate failed: {e}")
                QTimer.singleShot(0, lambda: QMessageBox.critical(
                    self, "Error", f"Failed: {e}"
                ))
        
        asyncio.ensure_future(do_mark())
    
    def set_selection_manager(self, selection_manager):
        """Connect to SelectionManager for selection changes."""
        self._selection_manager = selection_manager
        
        if hasattr(selection_manager, 'signals') and hasattr(selection_manager.signals, 'selection_changed'):
            selection_manager.signals.selection_changed.connect(self._on_selection_changed)
            logger.debug("SimilarItemsPanel connected to SelectionManager")
    
    def _on_selection_changed(self, file_ids: list, count: int, source: str):
        """Handle selection change - search for similar files."""
        if not file_ids:
            self._current_file_id = None
            self._list.clear()
            self._status_label.setText("No file selected")
            return
        
        # Use first selected file
        self._current_file_id = file_ids[0]
        asyncio.ensure_future(self._search_similar())
    
    def _on_refresh(self):
        """Refresh button clicked."""
        if self._current_file_id:
            asyncio.ensure_future(self._search_similar())
    
    async def _search_similar(self):
        """Search for similar files."""
        if not self._similarity_service or not self._current_file_id:
            return
        
        try:
            self._status_label.setText("Searching...")
            self._list.clear()
            
            from bson import ObjectId
            
            threshold = self._threshold_slider.value() / 100.0
            
            # Search via SimilarityService
            results = await self._similarity_service.find_similar(
                file_id=ObjectId(self._current_file_id),
                provider="clip",
                threshold=threshold,
                limit=10
            )
            
            if not results:
                self._status_label.setText("No similar files found")
                return
            
            # Populate list
            for result in results:
                file_record = result.get("file")
                score = result.get("score", 0)
                
                if file_record:
                    item = QListWidgetItem()
                    item.setText(f"{file_record.name}\nScore: {score:.2f}")
                    item.setData(Qt.ItemDataRole.UserRole, str(file_record._id))
                    item.setToolTip(file_record.path)
                    
                    # Try to load thumbnail
                    if self._thumbnail_service:
                        asyncio.ensure_future(
                            self._load_thumbnail(item, file_record._id)
                        )
                    
                    self._list.addItem(item)
            
            self._status_label.setText(f"Found {len(results)} similar files")
            
        except Exception as e:
            logger.error(f"Failed to search similar: {e}")
            self._status_label.setText(f"Error: {e}")
    
    async def _load_thumbnail(self, item: QListWidgetItem, file_id):
        """Load thumbnail for list item."""
        try:
            thumb_path = await self._thumbnail_service.get_or_create(file_id, size=128)
            if thumb_path and Path(thumb_path).exists():
                pixmap = QPixmap(str(thumb_path))
                if not pixmap.isNull():
                    from PySide6.QtGui import QIcon
                    item.setIcon(QIcon(pixmap))
        except Exception as e:
            logger.debug(f"Failed to load thumbnail: {e}")
    
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click - emit signal for navigation."""
        file_id = item.data(Qt.ItemDataRole.UserRole)
        if file_id:
            self.item_clicked.emit(file_id)
    
    def on_update(self, context=None):
        """Called when panel is updated."""
        pass
