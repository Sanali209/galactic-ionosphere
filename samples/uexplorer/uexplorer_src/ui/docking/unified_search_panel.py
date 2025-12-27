"""
UnifiedSearchPanel - Merged Search + Filter panel with auto-execute.

Combines text search, filters, and displays active filter summary.
Auto-rebuilds and executes search when any source changes.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QScrollArea,
    QFrame, QSlider, QToolButton
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QIcon
from loguru import logger


class FilterSummaryWidget(QWidget):
    """Displays active filters in compact collapsible format."""
    
    clear_all_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Header with count
        header = QHBoxLayout()
        self.toggle_btn = QToolButton()
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.toggled.connect(self._on_toggle)
        header.addWidget(self.toggle_btn)
        
        self.header_label = QLabel("Active Filters (0)")
        self.header_label.setStyleSheet("font-weight: bold; color: #aaa;")
        header.addWidget(self.header_label)
        
        header.addStretch()
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setFixedHeight(20)
        self.clear_btn.setStyleSheet("font-size: 10px;")
        self.clear_btn.clicked.connect(self.clear_all_clicked.emit)
        self.clear_btn.hide()
        header.addWidget(self.clear_btn)
        
        layout.addLayout(header)
        
        # Content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 4, 4, 4)
        self.content_layout.setSpacing(2)
        
        self.dir_label = QLabel()
        self.dir_label.setStyleSheet("color: #888; font-size: 11px;")
        self.dir_label.hide()
        self.content_layout.addWidget(self.dir_label)
        
        self.tag_label = QLabel()
        self.tag_label.setStyleSheet("color: #888; font-size: 11px;")
        self.tag_label.hide()
        self.content_layout.addWidget(self.tag_label)
        
        self.album_label = QLabel()
        self.album_label.setStyleSheet("color: #888; font-size: 11px;")
        self.album_label.hide()
        self.content_layout.addWidget(self.album_label)
        
        self.filter_label = QLabel()
        self.filter_label.setStyleSheet("color: #888; font-size: 11px;")
        self.filter_label.hide()
        self.content_layout.addWidget(self.filter_label)
        
        layout.addWidget(self.content)
    
    def _on_toggle(self, checked):
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(checked)
    
    def update_from_query(self, query):
        """Update display from UnifiedSearchQuery."""
        count = 0
        
        # Directories
        dir_parts = []
        if hasattr(query, 'directory_include') and query.directory_include:
            dir_parts.append(f"+{len(query.directory_include)}")
            count += 1
        if hasattr(query, 'directory_exclude') and query.directory_exclude:
            dir_parts.append(f"-{len(query.directory_exclude)}")
            count += 1
        if dir_parts:
            self.dir_label.setText(f"📁 Directories: {', '.join(dir_parts)}")
            self.dir_label.show()
        else:
            self.dir_label.hide()
        
        # Tags
        tag_parts = []
        if hasattr(query, 'tag_include') and query.tag_include:
            tag_parts.append(f"+{len(query.tag_include)}")
            count += 1
        if hasattr(query, 'tag_exclude') and query.tag_exclude:
            tag_parts.append(f"-{len(query.tag_exclude)}")
            count += 1
        if tag_parts:
            self.tag_label.setText(f"🏷️ Tags: {', '.join(tag_parts)}")
            self.tag_label.show()
        else:
            self.tag_label.hide()
        
        # Albums
        album_parts = []
        if hasattr(query, 'album_include') and query.album_include:
            album_parts.append(f"+{len(query.album_include)}")
            count += 1
        if hasattr(query, 'album_exclude') and query.album_exclude:
            album_parts.append(f"-{len(query.album_exclude)}")
            count += 1
        if album_parts:
            self.album_label.setText(f"📚 Albums: {', '.join(album_parts)}")
            self.album_label.show()
        else:
            self.album_label.hide()
        
        # Filters
        filter_parts = []
        if hasattr(query, 'filters') and query.filters:
            for k, v in query.filters.items():
                if v:
                    filter_parts.append(k)
                    count += 1
        if filter_parts:
            self.filter_label.setText(f"⚙️ Filters: {', '.join(filter_parts)}")
            self.filter_label.show()
        else:
            self.filter_label.hide()
        
        self.header_label.setText(f"Active Filters ({count})")
        self.clear_btn.setVisible(count > 0)


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
        self._debounce_timer.setInterval(300)
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
        
        # === Filter Summary ===
        self.filter_summary = FilterSummaryWidget()
        self.filter_summary.clear_all_clicked.connect(self._on_clear_all)
        layout.addWidget(self.filter_summary)
        
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
        
        return filters
    
    def _on_rating_changed(self, value):
        if value == 0:
            self.rating_label.setText("Any")
        else:
            self.rating_label.setText("★" * value)
        self._schedule_search()
    
    def _on_query_changed(self, query):
        """Update filter summary when query changes."""
        self.filter_summary.update_from_query(query)
    
    def _on_clear_all(self):
        """Clear all filters."""
        # Clear local
        self.search_input.clear()
        self.cb_images.setChecked(False)
        self.cb_videos.setChecked(False)
        self.cb_audio.setChecked(False)
        self.cb_docs.setChecked(False)
        self.rating_slider.setValue(0)
        
        # Clear query builder
        if self._query_builder:
            self._query_builder.clear_all()
