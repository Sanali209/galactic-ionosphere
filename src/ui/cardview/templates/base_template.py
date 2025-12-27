"""
BaseCardTemplate - Abstract base for card templates.

Provides common functionality for custom card templates.
"""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtGui import QPixmap

from src.ui.cardview.card_item_widget import CardItemWidget

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class BaseCardTemplate(CardItemWidget):
    """
    Abstract base class for card templates.
    
    Provides structure for creating custom card layouts.
    Subclass and implement build_content() and update_display().
    
    Features:
    - Standard layout structure
    - Thumbnail area
    - Title/subtitle labels
    - Selection styling
    
    Example:
        class MyCardTemplate(BaseCardTemplate):
            def build_content(self):
                self.thumbnail = QLabel()
                self.title = QLabel()
                self.layout.addWidget(self.thumbnail)
                self.layout.addWidget(self.title)
            
            def update_display(self):
                self.title.setText(self.data_context.title)
    """
    
    def __init__(self, parent: QWidget | None = None):
        """Initialize template."""
        super().__init__(parent)
    
    def _setup_base_ui(self):
        """Setup base structure."""
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Content area (to be filled by subclass)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(4)
        
        self.layout.addWidget(self.content_widget)
        
        # Build template content
        self.build_content()
        
        # Apply default style
        self._update_style()
    
    @abstractmethod
    def build_content(self):
        """
        Build template content.
        
        Override to create your card layout.
        Add widgets to self.content_layout.
        """
        pass
    
    @abstractmethod
    def update_display(self):
        """
        Update display from data_context.
        
        Override to populate widgets from self.data_context.
        """
        pass
    
    def clear_content(self):
        """Clear for recycling. Override if needed."""
        pass
    
    def _update_style(self):
        """Update selection style."""
        if self._selected:
            self.setStyleSheet("""
                BaseCardTemplate {
                    background-color: #e3f2fd;
                    border: 2px solid #2196f3;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                BaseCardTemplate {
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                }
                BaseCardTemplate:hover {
                    background-color: #fafafa;
                    border-color: #bdbdbd;
                }
            """)


class DefaultCardTemplate(BaseCardTemplate):
    """
    Default card template showing title and subtitle.
    """
    
    def build_content(self):
        """Build default layout."""
        # Thumbnail placeholder
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(120)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border-radius: 4px;
            }
        """)
        self.content_layout.addWidget(self.thumbnail_label)
        
        # Title
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
            }
        """)
        self.content_layout.addWidget(self.title_label)
        
        # Subtitle
        self.subtitle_label = QLabel()
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #666;
            }
        """)
        self.content_layout.addWidget(self.subtitle_label)
    
    def update_display(self):
        """Update from data_context."""
        if self._data_context:
            self.title_label.setText(self._data_context.title or "")
            self.subtitle_label.setText(self._data_context.subtitle or "")
    
    def clear_content(self):
        """Clear for recycling."""
        self.thumbnail_label.clear()
        self.title_label.clear()
        self.subtitle_label.clear()
