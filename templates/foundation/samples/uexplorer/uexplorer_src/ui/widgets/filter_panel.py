"""
UExplorer - Advanced Filter Panel

Dockable panel for complex filtering with tag selection,
file type checkboxes, rating slider, and date range.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QSlider, QGroupBox, QScrollArea, QComboBox,
    QFrame, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from loguru import logger


class FilterPanel(QWidget):
    """
    Advanced filter panel with multiple filter options.
    
    Features:
    - Tag selection with mode (Any/All/None)
    - File type checkboxes
    - Rating filter slider
    - Apply/Clear buttons
    """
    
    filters_applied = Signal()
    filters_cleared = Signal()
    
    def __init__(self, filter_manager=None, locator=None, parent=None):
        super().__init__(parent)
        self._filter_manager = filter_manager
        self._locator = locator
        self._selected_tags = set()
        
        self.setup_ui()
        
        logger.info("FilterPanel initialized")
    
    def setup_ui(self):
        """Setup UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # Title
        title = QLabel("Advanced Filters")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(title)
        
        # Scroll area for filter groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)
        
        # === File Types ===
        type_group = QGroupBox("File Types")
        type_layout = QVBoxLayout(type_group)
        
        self.type_checkboxes = {}
        file_types = [
            ("Images", "image"),
            ("Videos", "video"),
            ("Audio", "audio"),
            ("Documents", "document"),
            ("Raw Files", "raw"),
        ]
        
        for label, file_type in file_types:
            cb = QCheckBox(label)
            cb.setProperty("file_type", file_type)
            cb.stateChanged.connect(self._on_type_changed)
            self.type_checkboxes[file_type] = cb
            type_layout.addWidget(cb)
        
        scroll_layout.addWidget(type_group)
        
        # === Rating ===
        rating_group = QGroupBox("Minimum Rating")
        rating_layout = QVBoxLayout(rating_group)
        
        self.rating_slider = QSlider(Qt.Horizontal)
        self.rating_slider.setMinimum(0)
        self.rating_slider.setMaximum(5)
        self.rating_slider.setValue(0)
        self.rating_slider.setTickPosition(QSlider.TicksBelow)
        self.rating_slider.setTickInterval(1)
        self.rating_slider.valueChanged.connect(self._on_rating_changed)
        
        self.rating_label = QLabel("Any (0+)")
        self.rating_label.setAlignment(Qt.AlignCenter)
        
        rating_layout.addWidget(self.rating_slider)
        rating_layout.addWidget(self.rating_label)
        
        scroll_layout.addWidget(rating_group)
        
        # === Tags ===
        tags_group = QGroupBox("Tags")
        tags_layout = QVBoxLayout(tags_group)
        
        # Tag mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.tag_mode = QComboBox()
        self.tag_mode.addItems(["Any", "All", "None"])
        self.tag_mode.currentTextChanged.connect(self._on_tag_mode_changed)
        mode_layout.addWidget(self.tag_mode, 1)
        tags_layout.addLayout(mode_layout)
        
        # Tag list (placeholder - will be populated dynamically)
        self.tag_list_label = QLabel("Select tags from Tags panel")
        self.tag_list_label.setStyleSheet("color: #888888; font-style: italic;")
        tags_layout.addWidget(self.tag_list_label)
        
        self.selected_tags_label = QLabel("")
        self.selected_tags_label.setWordWrap(True)
        tags_layout.addWidget(self.selected_tags_label)
        
        scroll_layout.addWidget(tags_group)
        
        # === Advanced Filter Tree ===
        tree_group = QGroupBox("Advanced Query Builder")
        tree_layout = QVBoxLayout(tree_group)
        
        try:
            from uexplorer_src.ui.widgets.filter_tree_widget import FilterTreeWidget
            self.filter_tree = FilterTreeWidget()
            tree_layout.addWidget(self.filter_tree)
        except ImportError:
            self.filter_tree = None
            tree_layout.addWidget(QLabel("Filter tree not available"))
        
        scroll_layout.addWidget(tree_group)
        
        # Spacer
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)
        
        # === Action buttons ===
        btn_layout = QHBoxLayout()
        
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear)
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_filters)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #5a7aaa;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a8aba;
            }
        """)
        btn_layout.addWidget(self.btn_apply)
        
        main_layout.addLayout(btn_layout)
        
        # Apply theme
        self._apply_style()
    
    def _apply_style(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a7aaa;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QLabel {
                color: #cccccc;
            }
        """)
    
    def set_filter_manager(self, manager):
        """Set FilterManager reference."""
        self._filter_manager = manager
    
    def _on_type_changed(self, state):
        """Handle file type checkbox change."""
        pass  # Applied on "Apply" button
    
    def _on_rating_changed(self, value):
        """Handle rating slider change."""
        if value == 0:
            self.rating_label.setText("Any (0+)")
        else:
            stars = "★" * value
            self.rating_label.setText(f"{stars} ({value}+)")
    
    def _on_tag_mode_changed(self, mode_text):
        """Handle tag mode change."""
        pass  # Applied on "Apply" button
    
    def add_selected_tag(self, tag_id, tag_name):
        """Add tag to selection."""
        self._selected_tags.add(tag_id)
        self._update_selected_tags_label()
    
    def remove_selected_tag(self, tag_id):
        """Remove tag from selection."""
        self._selected_tags.discard(tag_id)
        self._update_selected_tags_label()
    
    def _update_selected_tags_label(self):
        """Update label showing selected tags."""
        count = len(self._selected_tags)
        if count == 0:
            self.selected_tags_label.setText("")
        else:
            self.selected_tags_label.setText(f"{count} tag(s) selected")
    
    def apply_filters(self):
        """Apply all filters to FilterManager."""
        if not self._filter_manager:
            logger.warning("No FilterManager connected")
            return
        
        # File types
        selected_types = [
            cb.property("file_type")
            for cb in self.type_checkboxes.values()
            if cb.isChecked()
        ]
        if selected_types:
            self._filter_manager.set_file_types(selected_types)
        else:
            self._filter_manager.clear_file_types()
        
        # Rating
        rating = self.rating_slider.value()
        self._filter_manager.set_rating_filter(rating)
        
        # Tags
        mode = self.tag_mode.currentText().lower()
        if self._selected_tags:
            self._filter_manager.set_tag_filter(list(self._selected_tags), mode)
        else:
            self._filter_manager.clear_tag_filter()
        
        self.filters_applied.emit()
        logger.info(f"Filters applied: types={selected_types}, rating>={rating}")
    
    def clear_all(self):
        """Clear all filter selections."""
        # Clear checkboxes
        for cb in self.type_checkboxes.values():
            cb.setChecked(False)
        
        # Reset rating
        self.rating_slider.setValue(0)
        
        # Clear tags
        self._selected_tags.clear()
        self._update_selected_tags_label()
        self.tag_mode.setCurrentIndex(0)
        
        # Clear FilterManager
        if self._filter_manager:
            self._filter_manager.clear_all()
        
        self.filters_cleared.emit()
        logger.info("All filters cleared")
