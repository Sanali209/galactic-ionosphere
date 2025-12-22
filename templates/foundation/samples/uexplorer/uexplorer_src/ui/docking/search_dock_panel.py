"""
UExplorer - Search Dock Panel

Dockable search panel with text/vector search toggle.
Combines search results with active filters from FilterPanel.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel,
    QPushButton, QButtonGroup, QToolButton, QFrame
)
from PySide6.QtCore import Qt, Signal
from loguru import logger


class SearchDockPanel(QWidget):
    """
    Dockable search panel.
    
    Features:
    - Text search input
    - Toggle: Text vs Vector (Similar) mode
    - Executes search combining with active filters
    
    Signals:
        search_requested: Emitted with (mode, query_text)
        results_ready: Emitted with search results
    """
    
    search_requested = Signal(str, str, list)  # mode, query_text, fields
    results_ready = Signal(list)
    
    def __init__(self, filter_manager=None, locator=None, parent=None):
        super().__init__(parent)
        self._filter_manager = filter_manager
        self._locator = locator
        self._mode = "text"  # "text" or "vector"
        
        self.setup_ui()
        
        logger.info("SearchDockPanel initialized")
    
    def setup_ui(self):
        """Build UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("🔍 Search")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Mode toggle
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(4)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        
        self.btn_text = QToolButton()
        self.btn_text.setText("📝 Text")
        self.btn_text.setCheckable(True)
        self.btn_text.setChecked(True)
        self.btn_text.setToolTip("Search by file name")
        self.btn_text.setProperty("mode", "text")
        self.mode_group.addButton(self.btn_text)
        mode_layout.addWidget(self.btn_text)
        
        self.btn_vector = QToolButton()
        self.btn_vector.setText("🎯 Similar")
        self.btn_vector.setCheckable(True)
        self.btn_vector.setToolTip("Find visually similar images (AI)")
        self.btn_vector.setProperty("mode", "vector")
        self.mode_group.addButton(self.btn_vector)
        mode_layout.addWidget(self.btn_vector)
        
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #4a4a4a;")
        layout.addWidget(sep)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self._on_search)
        layout.addWidget(self.search_input)
        
        # Mode description
        self.mode_label = QLabel("Search in file names")
        self.mode_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.mode_label)
        
        # Field selection checkboxes
        from PySide6.QtWidgets import QCheckBox, QGridLayout
        
        fields_label = QLabel("Fields:")
        fields_label.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-top: 8px;")
        layout.addWidget(fields_label)
        
        fields_grid = QGridLayout()
        fields_grid.setSpacing(4)
        
        self.chk_name = QCheckBox("Name")
        self.chk_name.setChecked(True)
        self.chk_name.setStyleSheet("color: #cccccc;")
        fields_grid.addWidget(self.chk_name, 0, 0)
        
        self.chk_path = QCheckBox("Path")
        self.chk_path.setChecked(True)
        self.chk_path.setStyleSheet("color: #cccccc;")
        fields_grid.addWidget(self.chk_path, 0, 1)
        
        self.chk_description = QCheckBox("Description")
        self.chk_description.setStyleSheet("color: #cccccc;")
        fields_grid.addWidget(self.chk_description, 1, 0)
        
        self.chk_tags = QCheckBox("Tags")
        self.chk_tags.setStyleSheet("color: #cccccc;")
        fields_grid.addWidget(self.chk_tags, 1, 1)
        
        layout.addLayout(fields_grid)
        
        # Separator before search button
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: #4a4a4a;")
        layout.addWidget(sep2)
        
        # Search button
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self._on_search)
        layout.addWidget(self.btn_search)
        
        # Filter indicator
        self.filter_label = QLabel("")
        self.filter_label.setStyleSheet("color: #5a9aca; font-size: 11px;")
        layout.addWidget(self.filter_label)
        
        layout.addStretch()
        
        self._apply_style()
        self._update_filter_label()
    
    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #5a8aca;
            }
            QToolButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
            }
            QToolButton:checked {
                background-color: #5a7aaa;
                color: white;
                border-color: #7a9aca;
            }
            QPushButton {
                background-color: #5a7aaa;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a8aba;
            }
        """)
    
    def set_filter_manager(self, manager):
        """Set filter manager reference."""
        self._filter_manager = manager
        if manager:
            manager.filters_changed.connect(self._update_filter_label)
    
    def _on_mode_changed(self, button):
        """Handle mode toggle."""
        mode = button.property("mode")
        self._mode = mode
        
        if mode == "text":
            self.search_input.setPlaceholderText("Enter search query...")
            self.mode_label.setText("Search in file names")
        else:
            self.search_input.setPlaceholderText("Describe what you're looking for...")
            self.mode_label.setText("Find visually similar images using AI")
        
        logger.debug(f"Search mode: {mode}")
    
    def get_selected_fields(self) -> list:
        """Get list of selected search fields."""
        fields = []
        if self.chk_name.isChecked():
            fields.append("name")
        if self.chk_path.isChecked():
            fields.append("path")
        if self.chk_description.isChecked():
            fields.append("description")
        if self.chk_tags.isChecked():
            fields.append("tags")
        return fields if fields else ["name"]  # Default to name
    
    def _on_search(self):
        """Execute search (empty text = filter-only mode)."""
        query = self.search_input.text().strip()
        # Allow empty text search (filter-only mode)
        
        fields = self.get_selected_fields()
        self.search_requested.emit(self._mode, query, fields)
        logger.info(f"Search requested: mode={self._mode}, query='{query}', fields={fields}")
    
    def _update_filter_label(self):
        """Update filter indicator."""
        if self._filter_manager:
            count = self._filter_manager.active_filter_count()
            if count > 0:
                self.filter_label.setText(f"+ {count} active filter(s)")
            else:
                self.filter_label.setText("")
        else:
            self.filter_label.setText("")
    
    def get_combined_query(self):
        """
        Get search query combined with active filters.
        
        Returns:
            Q expression combining search and filters
        """
        from src.ucorefs.query.builder import Q
        
        query_terms = []
        
        # Add text/vector search
        search_text = self.search_input.text().strip()
        if search_text:
            if self._mode == "text":
                query_terms.append(Q.name_contains(search_text))
        
        # Add active filters
        if self._filter_manager:
            filter_q = self._filter_manager.get_query()
            if filter_q:
                query_terms.append(filter_q)
        
        # Combine with AND
        if len(query_terms) == 0:
            return None
        elif len(query_terms) == 1:
            return query_terms[0]
        else:
            return Q.AND(*query_terms)
