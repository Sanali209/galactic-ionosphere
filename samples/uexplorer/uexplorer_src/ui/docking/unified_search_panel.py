"""
UnifiedSearchPanel - Merged Search + Filter panel with auto-execute.

Combines text search, filters, and displays active filter badges.
Auto-rebuilds and executes search when any source changes.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QScrollArea,
    QFrame, QSlider, QToolButton
)
from PySide6.QtCore import Signal, QTimer, Qt, QTimer
from PySide6.QtGui import QIcon
from loguru import logger


class UnifiedSearchPanel(QWidget):
    """
    Unified Search + Filter panel with auto-execute.
    
    Features:
    - Text search input with mode selector
    - Field checkboxes for text search
    - Filter summary showing active filters from all panels
    - Built-in file type and rating filters
    - Auto-executes search with 300ms debounce
    
    Signals:
        search_requested(mode, text, fields): Manual search triggered
    """
    
    search_requested = Signal(str, str, list)  # mode, text, fields
    
    def __init__(self, locator=None, parent=None):
        super().__init__(parent)
        self._locator = locator
        self._query_builder = None
        
        # Debounce timer
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(2000)  # 2 seconds - avoid CLIP spam
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)
        
        self._setup_ui()
        self._apply_style()
        
        logger.info("UnifiedSearchPanel initialized")
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # === Search Input ===
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout(search_group)
        
        # Mode + Input row
        input_row = QHBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Text", "Semantic"])
        self.mode_combo.setToolTip("Text: regex match | Semantic: AI similarity")
        self.mode_combo.currentIndexChanged.connect(self._schedule_search)
        input_row.addWidget(self.mode_combo)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search files...")
        self.search_input.textChanged.connect(self._schedule_search)
        self.search_input.returnPressed.connect(self._execute_now)
        input_row.addWidget(self.search_input, 1)
        
        search_layout.addLayout(input_row)
        
        # Field checkboxes
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Fields:"))
        
        self.cb_name = QCheckBox("Name")
        self.cb_name.setChecked(True)
        self.cb_name.stateChanged.connect(self._schedule_search)
        fields_row.addWidget(self.cb_name)
        
        self.cb_path = QCheckBox("Path")
        self.cb_path.stateChanged.connect(self._schedule_search)
        fields_row.addWidget(self.cb_path)
        
        self.cb_desc = QCheckBox("Desc")
        self.cb_desc.stateChanged.connect(self._schedule_search)
        fields_row.addWidget(self.cb_desc)
        
        self.cb_ai = QCheckBox("AI")
        self.cb_ai.stateChanged.connect(self._schedule_search)
        fields_row.addWidget(self.cb_ai)
        
        fields_row.addStretch()
        search_layout.addLayout(fields_row)
        
        layout.addWidget(search_group)
        
        # === Active Filters (Badge Display) ===
        try:
            from uexplorer_src.ui.widgets.active_filters_widget import ActiveFiltersWidget
            self.active_filters = ActiveFiltersWidget()
            self.active_filters.clear_all_requested.connect(self._on_clear_all)
            self.active_filters.filter_removed.connect(self._on_filter_badge_removed)
            layout.addWidget(self.active_filters)
        except ImportError as e:
            logger.error(f"Failed to import ActiveFiltersWidget: {e}")
            # Fallback to simple label
            self.active_filters = QLabel("Filters widget unavailable")
            self.active_filters.setStyleSheet("color: #888;")
            layout.addWidget(self.active_filters)
        
        # === Quick Filters ===
        filter_group = QGroupBox("Quick Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        # File type row
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        
        self.cb_images = QCheckBox("🖼️")
        self.cb_images.setToolTip("Images")
        self.cb_images.stateChanged.connect(self._schedule_search)
        type_row.addWidget(self.cb_images)
        
        self.cb_videos = QCheckBox("🎬")
        self.cb_videos.setToolTip("Videos")
        self.cb_videos.stateChanged.connect(self._schedule_search)
        type_row.addWidget(self.cb_videos)
        
        self.cb_audio = QCheckBox("🎵")
        self.cb_audio.setToolTip("Audio")
        self.cb_audio.stateChanged.connect(self._schedule_search)
        type_row.addWidget(self.cb_audio)
        
        self.cb_docs = QCheckBox("📄")
        self.cb_docs.setToolTip("Documents")
        self.cb_docs.stateChanged.connect(self._schedule_search)
        type_row.addWidget(self.cb_docs)
        
        type_row.addStretch()
        filter_layout.addLayout(type_row)
        
        # Rating row
        rating_row = QHBoxLayout()
        rating_row.addWidget(QLabel("Rating:"))
        
        self.rating_slider = QSlider(Qt.Orientation.Horizontal)
        self.rating_slider.setRange(0, 5)
        self.rating_slider.setValue(0)
        self.rating_slider.valueChanged.connect(self._on_rating_changed)
        rating_row.addWidget(self.rating_slider)
        
        self.rating_label = QLabel("Any")
        self.rating_label.setMinimumWidth(40)
        rating_row.addWidget(self.rating_label)
        
        # Unrated checkbox
        self.cb_unrated = QCheckBox("Unrated")
        self.cb_unrated.setToolTip("Show only files with no rating")
        self.cb_unrated.stateChanged.connect(self._on_unrated_changed)
        rating_row.addWidget(self.cb_unrated)
        
        filter_layout.addLayout(rating_row)
        
        layout.addWidget(filter_group)
        
        layout.addStretch()
    
    def _apply_style(self):
        self.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QCheckBox {
                color: #cccccc;
            }
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a7aaa;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
    
    def set_query_builder(self, query_builder):
        """Connect to UnifiedQueryBuilder for updates."""
        self._query_builder = query_builder
        if query_builder:
            query_builder.query_changed.connect(self._on_query_changed)
    
    def _schedule_search(self):
        """Schedule search with debounce."""
        self._debounce_timer.start()
    
    def _on_debounce_timeout(self):
        """Execute search after debounce."""
        self._execute_now()
    
    def _execute_now(self):
        """Execute search immediately."""
        mode = "text" if self.mode_combo.currentIndex() == 0 else "semantic"
        text = self.search_input.text()
        fields = self._get_selected_fields()
        
        # Update query builder with our values
        if self._query_builder:
            self._query_builder.set_text_search(mode, text, fields)
            self._query_builder.set_filters(self._get_local_filters())
        
        self.search_requested.emit(mode, text, fields)
    
    def _get_selected_fields(self) -> list:
        fields = []
        if self.cb_name.isChecked():
            fields.append("name")
        if self.cb_path.isChecked():
            fields.append("path")
        if self.cb_desc.isChecked():
            fields.append("description")
        if self.cb_ai.isChecked():
            fields.append("ai_description")
        return fields or ["name"]
    
    def _get_local_filters(self) -> dict:
        """Get filters from this panel."""
        filters = {}
        
        # File types
        types = []
        if self.cb_images.isChecked():
            types.append("image")
        if self.cb_videos.isChecked():
            types.append("video")
        if self.cb_audio.isChecked():
            types.append("audio")
        if self.cb_docs.isChecked():
            types.append("document")
        if types:
            filters["file_type"] = types
        
        # Rating
        rating = self.rating_slider.value()
        if rating > 0:
            filters["rating"] = rating
        
        # Unrated filter (overrides rating slider)
        if self.cb_unrated.isChecked():
            filters["unrated"] = True  # Special flag for unrated files
        
        return filters
    
    def _on_rating_changed(self, value):
        # Uncheck unrated when slider moves
        if value > 0 and self.cb_unrated.isChecked():
            self.cb_unrated.setChecked(False)
        
        if value == 0:
            self.rating_label.setText("Any")
        else:
            self.rating_label.setText("★" * value)
        self._schedule_search()
    
    def _on_unrated_changed(self, state):
        # Reset slider when unrated is checked
        if state and self.rating_slider.value() > 0:
            self.rating_slider.setValue(0)
        self._schedule_search()
    
    def _on_query_changed(self, query):
        """Update filter badges when query changes."""
        if not hasattr(self, 'active_filters'):
            return
        
        # Skip if active_filters is just a fallback label
        if isinstance(self.active_filters, QLabel):
            return
        
        # Clear all existing badges
        self.active_filters.clear_all()
        
        # Add directory badges
        if hasattr(query, 'directory_include') and query.directory_include:
            for path in query.directory_include:
                # Resolve directory name from database
                display_name = self._get_directory_name(path)
                self.active_filters.add_badge(path, display_name, "directory", include=True)
        
        if hasattr(query, 'directory_exclude') and query.directory_exclude:
            for path in query.directory_exclude:
                # Resolve directory name from database
                display_name = self._get_directory_name(path)
                self.active_filters.add_badge(path, display_name, "directory", include=False)
        
        # === TAGS - Prefer refs (no DB query!), fallback to legacy ===
        if hasattr(query, 'tag_refs_include') and query.tag_refs_include:
            # NEW WAY: Use TagRef objects - names already available!
            for tag_ref in query.tag_refs_include:
                self.active_filters.add_badge(
                    tag_ref.to_id_str(),
                    tag_ref.name,  # ← NO database query!
                    "tag",
                    include=True
                )
        elif hasattr(query, 'tag_include') and query.tag_include:
            # FALLBACK: Legacy IDs - need database query
            for tag_id in query.tag_include:
                tag_name = self._get_tag_name(tag_id)
                self.active_filters.add_badge(tag_id, tag_name, "tag", include=True)
        
        if hasattr(query, 'tag_refs_exclude') and query.tag_refs_exclude:
            for tag_ref in query.tag_refs_exclude:
                self.active_filters.add_badge(tag_ref.to_id_str(), tag_ref.name, "tag", include=False)
        elif hasattr(query, 'tag_exclude') and query.tag_exclude:
            for tag_id in query.tag_exclude:
                tag_name = self._get_tag_name(tag_id)
                self.active_filters.add_badge(tag_id, tag_name, "tag", include=False)
        
        # === ALBUMS - Prefer refs (no DB query!), fallback to legacy ===
        if hasattr(query, 'album_refs_include') and query.album_refs_include:
            # NEW WAY: Use AlbumRef objects - names already available!
            for album_ref in query.album_refs_include:
                self.active_filters.add_badge(
                    album_ref.to_id_str(),
                    album_ref.name,  # ← NO database query!
                    "album",
                    include=True
                )
        elif hasattr(query, 'album_include') and query.album_include:
            # FALLBACK: Legacy IDs - need database query
            for album_id in query.album_include:
                album_name = self._get_album_name(album_id)
                self.active_filters.add_badge(album_id, album_name, "album", include=True)
        
        if hasattr(query, 'album_refs_exclude') and query.album_refs_exclude:
            for album_ref in query.album_refs_exclude:
                self.active_filters.add_badge(album_ref.to_id_str(), album_ref.name, "album", include=False)
        elif hasattr(query, 'album_exclude') and query.album_exclude:
            for album_id in query.album_exclude:
                album_name = self._get_album_name(album_id)
                self.active_filters.add_badge(album_id, album_name, "album", include=False)
    
    def _get_directory_name(self, path: str) -> str:
        """Resolve directory path to display name."""
        # Directories use paths, not IDs - extract folder name
        from pathlib import Path
        try:
            folder_name = Path(path).name
            return folder_name if folder_name else path
        except Exception as e:
            logger.debug(f"Failed to extract directory name from path: {e}")
            return path
    
    def _get_tag_name(self, tag_id: str) -> str:
        """Resolve tag ID to display name."""
        logger.debug(f"Resolving tag name for ID: {tag_id}")
        
        try:
            from bson import ObjectId
            from pymongo import MongoClient
            
            # Get MongoDB connection info from config
            if self._locator and hasattr(self._locator, 'config'):
                config = self._locator.config
                if hasattr(config, 'data') and hasattr(config.data, 'mongo'):
                    host = config.data.mongo.host
                    port = config.data.mongo.port
                    db_name = config.data.mongo.database_name
                else:
                    host, port, db_name = 'localhost', 27017, 'app_db'
            else:
                host, port, db_name = 'localhost', 27017, 'app_db'
            
            # Create synchronous connection for this query
            sync_client = MongoClient(f"mongodb://{host}:{port}")
            sync_db = sync_client[db_name]
            
            try:
                tag_obj_id = ObjectId(tag_id)
                tag_doc = sync_db.tags.find_one({"_id": tag_obj_id})
                
                if tag_doc and 'name' in tag_doc:
                    logger.debug(f"Found tag name: {tag_doc['name']}")
                    return tag_doc['name']
                else:
                    logger.warning(f"Tag not found: {tag_id}")
            finally:
                sync_client.close()
            
            return tag_id[:8]
        except Exception as e:
            logger.error(f"Failed to resolve tag name: {e}")
            return tag_id[:8]
    
    def _get_album_name(self, album_id: str) -> str:
        """Resolve album ID to display name."""
        logger.debug(f"Resolving album name for ID: {album_id}")
        
        try:
            from bson import ObjectId
            from pymongo import MongoClient
            
            # Get MongoDB connection info from config
            if self._locator and hasattr(self._locator, 'config'):
                config = self._locator.config
                if hasattr(config, 'data') and hasattr(config.data, 'mongo'):
                    host = config.data.mongo.host
                    port = config.data.mongo.port
                    db_name = config.data.mongo.database_name
                else:
                    host, port, db_name = 'localhost', 27017, 'app_db'
            else:
                host, port, db_name = 'localhost', 27017, 'app_db'
            
            # Create synchronous connection for this query
            sync_client = MongoClient(f"mongodb://{host}:{port}")
            sync_db = sync_client[db_name]
            
            try:
                album_obj_id = ObjectId(album_id)
                album_doc = sync_db.albums.find_one({"_id": album_obj_id})
                
                if album_doc and 'name' in album_doc:
                    logger.debug(f"Found album name: {album_doc['name']}")
                    return album_doc['name']
                else:
                    logger.warning(f"Album not found: {album_id}")
            finally:
                sync_client.close()
            
            return album_id[:8]
        except Exception as e:
            logger.error(f"Failed to resolve album name: {e}")
            return album_id[:8]
    
    def _on_filter_badge_removed(self, filter_type: str, filter_id: str):
        """Handle filter badge removal - update query builder."""
        if not self._query_builder:
            logger.warning("No query builder connected")
            return
        
        logger.debug(f"Badge removed: {filter_type} - {filter_id}")
        
        # Get current query
        query = self._query_builder.get_current_query()
        
        # Remove from appropriate list
        if filter_type == "tag":
            if filter_id in query.tag_include:
                query.tag_include.remove(filter_id)
            if filter_id in query.tag_exclude:
                query.tag_exclude.remove(filter_id)
        elif filter_type == "album":
            if filter_id in query.album_include:
                query.album_include.remove(filter_id)
            if filter_id in query.album_exclude:
                query.album_exclude.remove(filter_id)
        elif filter_type == "directory":
            if filter_id in query.directory_include:
                query.directory_include.remove(filter_id)
            if filter_id in query.directory_exclude:
                query.directory_exclude.remove(filter_id)
        
        # Trigger query update
        self._query_builder._emit_query()
    
    def _on_clear_all(self):
        """Clear all filters."""
        # Clear local
        self.search_input.clear()
        self.cb_images.setChecked(False)
        self.cb_videos.setChecked(False)
        self.cb_audio.setChecked(False)
        self.cb_docs.setChecked(False)
        self.rating_slider.setValue(0)
        self.cb_unrated.setChecked(False)
        
        # Clear query builder
        if self._query_builder:
            self._query_builder.clear_all()
