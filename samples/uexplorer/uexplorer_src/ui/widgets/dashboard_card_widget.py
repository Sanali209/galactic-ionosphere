"""
DashboardCardWidget - Specialized card widget for Dashboard stats.

Renders:
- Stat: Large value + Label + Icon
- Progress: Title + Progress Bar + Details
- Action: Button-like appearance
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QFrame, QStyle
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QFont, QColor

class DashboardCardWidget(QFrame):
    """
    Widget for dashboard items (Stats, Progress, Actions).
    Standalone widget not dependent on CardView item logic.
    """
    
    # Signals matching CardItemWidget interface for compatibility
    clicked = Signal(str)
    double_clicked = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._type = "stat" 
        self._data_context = None
        
        # UI Components
        self._icon_label = QLabel()
        self._title_label = QLabel()
        self._value_label = QLabel()
        self._subtitle_label = QLabel()
        self._progress_bar = QProgressBar()
        
        self._setup_layout()
        self._apply_styles()
        
    def _setup_layout(self):
        """Build card layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # Header (Icon + Title)
        header_layout = QHBoxLayout()
        header_layout.addWidget(self._icon_label)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Content Area
        layout.addStretch()
        layout.addWidget(self._value_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._subtitle_label)
        layout.addStretch()
        
        # Initial visibility
        self._progress_bar.hide()
        self._value_label.show()

    def _apply_styles(self):
        """Apply styling."""
        self.setStyleSheet("""
            DashboardCardWidget {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            DashboardCardWidget:hover {
                border-color: #2196F3;
                background-color: #f8fbff;
            }
        """)
        
        self._title_label.setStyleSheet("color: #666; font-size: 12px; font-weight: bold;")
        self._icon_label.setStyleSheet("font-size: 18px;")
        self._value_label.setStyleSheet("color: #333; font-size: 24px; font-weight: bold;")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._subtitle_label.setStyleSheet("color: #888; font-size: 11px;")
        
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #eee;
                border-radius: 4px;
                height: 8px;
            }
            QProgressBar::chunk {
                background: #2196F3;
                border-radius: 4px;
            }
        """)
        self._progress_bar.setTextVisible(False)
    
    def bind_data(self, item):
        """Bind data from CardItem."""
        self._data_context = item
        data = item.data or {}
        self._type = getattr(item, 'item_type', 'stat')
        
        # Icon Mapping
        icon_map = {
            "library-books": "📚",
            "label": "🏷️",
            "photo-album": "🖼️",
            "pipeline": "⚙️",
            "share-variant": "🔗",
            "database-check": "✅",
            "calculator": "🧮",
            "circle-small": "•"
        }
        
        icon_name = data.get("icon", "circle-small")
        self._icon_label.setText(icon_map.get(icon_name, "•"))
        
        # Title & Subtitle
        self._title_label.setText(item.title)
        self._subtitle_label.setText(item.subtitle or "")
        
        # Config based on type
        if self._type == "stat":
            self._value_label.setText(str(data.get("value", "0")))
            self._value_label.show()
            self._progress_bar.hide()
            self.setStyleSheet(self.styleSheet().replace("border-color: #2196F3;", "border-color: #e0e0e0;"))
            
        elif self._type == "progress":
            current = data.get("current", 0)
            total = data.get("total", 100)
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._progress_bar.show()
            self._value_label.hide()
            
        elif self._type == "action":
            self._value_label.hide()
            self._progress_bar.hide()
            self._title_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")
            self.setStyleSheet(self.styleSheet().replace("#ffffff", "#f0f8ff")) # Light blue bg for actions

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._data_context:
                self.clicked.emit(self._data_context.id)
        super().mousePressEvent(event)
    
    # Compatibility method for CardItemWidget interface if needed elsewhere
    def set_size(self, w, h):
        self.setFixedSize(w, h)
