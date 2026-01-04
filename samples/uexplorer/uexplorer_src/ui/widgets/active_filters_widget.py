"""
Active Filters Widget - Display container for filter badges.

Shows all active filters grouped by category with flow layout.
Replaces FilterSummaryWidget with badge-based display.
"""

from typing import Dict, List, Tuple, Optional, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QLayout, QSizePolicy
)
from PySide6.QtCore import Signal, Qt, QRect, QSize, QPoint
from loguru import logger

from uexplorer_src.ui.widgets.filter_badge import FilterBadge


class QFlowLayout(QLayout):
    """Flow layout that wraps widgets to new lines when they exceed container width."""
    
    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0, spacing: int = -1) -> None:
        super().__init__(parent)
        self._items: List[Any] = []
        self._spacing: int = spacing
        
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
    
    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)
    
    def addItem(self, item):
        """Add item to layout."""
        self._items.append(item)
    
    def count(self):
        """Return number of items in layout."""
        return len(self._items)
    
    def itemAt(self, index):
        """Get item at index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None
    
    def takeAt(self, index):
        """Remove and return item at index."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None
    
    def expandingDirections(self):
        """Layout expands horizontally."""
        return Qt.Orientation(0)
    
    def hasHeightForWidth(self):
        """Layout height depends on width (for wrapping)."""
        return True
    
    def heightForWidth(self, width):
        """Calculate height needed for given width."""
        height = self._do_layout(QRect(0, 0, width, 0), True)
        return height
    
    def setGeometry(self, rect):
        """Position all widgets in the layout."""
        super().setGeometry(rect)
        self._do_layout(rect, False)
    
    def sizeHint(self):
        """Return preferred size."""
        return self.minimumSize()
    
    def minimumSize(self):
        """Calculate minimum size needed."""
        size = QSize()
        
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), 
                     margins.top() + margins.bottom())
        return size
    
    def _do_layout(self, rect, test_only):
        """
        Layout widgets in flow pattern.
        
        Args:
            rect: Rectangle to layout within
            test_only: If True, only calculate height without positioning
            
        Returns:
            Height needed for layout
        """
        left = rect.x()
        top = rect.y()
        line_height = 0
        
        spacing = self.spacing()
        if spacing == -1:
            spacing = 6  # Default spacing
        
        x = left
        y = top
        
        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue
                
            space_x = spacing
            space_y = spacing
            
            next_x = x + item.sizeHint().width() + space_x
            
            # Check if we need to wrap to next line
            if next_x - space_x > rect.right() and line_height > 0:
                x = left
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        
        return y + line_height - rect.y()
    
    def spacing(self):
        """Get spacing between items."""
        if self._spacing >= 0:
            return self._spacing
        else:
            # Get default spacing from parent
            parent = self.parent()
            if parent:
                return parent.style().layoutSpacing(
                    QSizePolicy.ControlType.PushButton,
                    QSizePolicy.ControlType.PushButton,
                    Qt.Orientation.Horizontal
                )
            return 6  # Fallback default


class ActiveFiltersWidget(QWidget):
    """
    Container displaying active filters as removable badges.
    
    Organizes badges by category (directories, tags, albums) with
    collapsible sections.
    
    Signals:
        filter_removed(filter_type, filter_id): Badge X button clicked
        clear_all_requested(): Clear All button clicked
    """
    
    filter_removed = Signal(str, str)  # (filter_type, filter_id)
    clear_all_requested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Storage for badges by category
        self._badges: Dict[str, List[Tuple[str, FilterBadge]]] = {
            "directory": [],
            "tag": [],
            "album": [],
            "detection": []
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the widget UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)
        
        # Header with "Active Filters" title and Clear All button
        header = QHBoxLayout()
        
        self.title_label = QLabel("Active Filters (0)")
        self.title_label.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 13px;")
        header.addWidget(self.title_label)
        
        header.addStretch()
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setFixedHeight(22)
        self.clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a5a5a;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #6a6a6a;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """)
        self.clear_all_btn.clicked.connect(self.clear_all_requested.emit)
        self.clear_all_btn.hide()
        header.addWidget(self.clear_all_btn)
        
        main_layout.addLayout(header)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #4a4a4a;")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)
        
        # Scroll area for badge sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(250)  # 3x typical height for better visibility
        scroll.setStyleSheet("""
            QScrollArea { 
                border: none; 
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
            }
        """)
        
        scroll_content = QWidget()
        self.sections_layout = QVBoxLayout(scroll_content)
        self.sections_layout.setContentsMargins(0, 0, 0, 0)
        self.sections_layout.setSpacing(12)
        
        # Create sections for each filter type
        self.directory_section = self._create_section("📁 Directories")
        self.tag_section = self._create_section("🏷️ Tags")
        self.album_section = self._create_section("📚 Albums")
        self.detection_section = self._create_section("👁️ Detections")
        
        self.sections_layout.addWidget(self.directory_section)
        self.sections_layout.addWidget(self.tag_section)
        self.sections_layout.addWidget(self.album_section)
        self.sections_layout.addWidget(self.detection_section)
        self.sections_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
    
    def _create_section(self, title: str) -> QWidget:
        """Create a collapsible section for a filter category."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Section title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Badge container with flow layout
        badge_container = QWidget()
        badge_container.setObjectName(f"{title}_badges")
        
        # Use proper wrapping flow layout
        badge_layout = QFlowLayout(spacing=6)
        badge_container.setLayout(badge_layout)
        layout.addWidget(badge_container)
        
        section.hide()  # Hidden until badges added
        return section
    
    def add_badge(self, filter_id: str, text: str, filter_type: str, include: bool = True):
        """
        Add a filter badge to the appropriate section.
        
        Args:
            filter_id: Unique identifier for this filter (tag_id, path, album_id)
            text: Display text for the badge
            filter_type: "tag", "directory", "album", "detection"
            include: True for include (+), False for exclude (-)
        """
        # Create badge
        badge = FilterBadge(text, filter_type, include)
        badge.removed.connect(lambda: self._on_badge_removed(filter_id, filter_type))
        
        # Add to appropriate section
        section_map = {
            "directory": self.directory_section,
            "tag": self.tag_section,
            "album": self.album_section,
            "detection": self.detection_section
        }
        
        section = section_map.get(filter_type)
        if not section:
            logger.warning(f"Unknown filter type: {filter_type}")
            return
        
        # Find badge container in section
        badge_container = section.findChild(QWidget, f"{section.findChild(QLabel).text()}_badges")
        if badge_container:
            badge_container.layout().addWidget(badge)
            section.show()
        
        # Store badge reference
        self._badges[filter_type].append((filter_id, badge))
        
        self._update_count()
    
    def remove_badge(self, filter_id: str, filter_type: str):
        """Remove a specific badge."""
        badges = self._badges.get(filter_type, [])
        
        for i, (stored_id, badge) in enumerate(badges):
            if stored_id == filter_id:
                # Remove widget
                badge.setParent(None)
                badge.deleteLater()
                
                # Remove from list
                del self._badges[filter_type][i]
                
                # Hide section if empty
                if not self._badges[filter_type]:
                    section_map = {
                        "directory": self.directory_section,
                        "tag": self.tag_section,
                        "album": self.album_section,
                        "detection": self.detection_section
                    }
                    if filter_type in section_map:
                        section_map[filter_type].hide()
                
                break
        
        self._update_count()
    
    def clear_all(self):
        """Remove all badges from all sections."""
        for filter_type in ["directory", "tag", "album", "detection"]:
            # Remove all badges
            for filter_id, badge in self._badges[filter_type]:
                badge.setParent(None)
                badge.deleteLater()
            
            self._badges[filter_type].clear()
            
            # Hide section
            section_map = {
                "directory": self.directory_section,
                "tag": self.tag_section,
                "album": self.album_section,
                "detection": self.detection_section
            }
            if filter_type in section_map:
                section_map[filter_type].hide()
        
        self._update_count()
    
    def clear_section(self, filter_type: str):
        """Clear all badges from a specific section."""
        if filter_type not in self._badges:
            return
        
        for filter_id, badge in self._badges[filter_type]:
            badge.setParent(None)
            badge.deleteLater()
        
        self._badges[filter_type].clear()
        
        # Hide section
        section_map = {
            "directory": self.directory_section,
            "tag": self.tag_section,
            "album": self.album_section,
            "detection": self.detection_section
        }
        section_map[filter_type].hide()
        
        self._update_count()
    
    def _on_badge_removed(self, filter_id: str, filter_type: str):
        """Handle badge removal."""
        self.remove_badge(filter_id, filter_type)
        self.filter_removed.emit(filter_type, filter_id)
    
    def _update_count(self):
        """Update title with total badge count."""
        total = sum(len(badges) for badges in self._badges.values())
        
        self.title_label.setText(f"Active Filters ({total})")
        self.clear_all_btn.setVisible(total > 0)
    
    def get_badge_count(self) -> int:
        """Get total number of active badges."""
        return sum(len(badges) for badges in self._badges.values())
