"""
DocumentCardTemplate - Card template for document items.

Shows file icon with title and metadata.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QWidget

from src.ui.cardview.templates.base_template import BaseCardTemplate


class DocumentCardTemplate(BaseCardTemplate):
    """
    Card template for documents and files.
    
    Features:
    - Large file type icon
    - Title and extension
    - Size and date info
    """
    
    # File type icons (unicode)
    ICONS = {
        "pdf": "📄",
        "doc": "📝",
        "docx": "📝",
        "xls": "📊",
        "xlsx": "📊",
        "ppt": "📽️",
        "pptx": "📽️",
        "txt": "📃",
        "zip": "📦",
        "rar": "📦",
        "default": "📁",
    }
    
    def build_content(self):
        """Build document card layout."""
        # Main row layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # Icon
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(48, 48)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(self.icon_label)
        
        # Info column
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        
        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
            }
        """)
        self.title_label.setWordWrap(True)
        info_layout.addWidget(self.title_label)
        
        # Subtitle (size, date)
        self.subtitle_label = QLabel()
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #888;
            }
        """)
        info_layout.addWidget(self.subtitle_label)
        
        info_layout.addStretch()
        main_layout.addWidget(info_widget, 1)
        
        self.content_layout.addLayout(main_layout)
    
    def update_display(self):
        """Update from data_context."""
        if not self._data_context:
            return
        
        # Title
        self.title_label.setText(self._data_context.title or "")
        
        # Subtitle
        self.subtitle_label.setText(self._data_context.subtitle or "")
        
        # Icon based on type or extension
        ext = ""
        if self._data_context.title:
            parts = self._data_context.title.rsplit(".", 1)
            if len(parts) > 1:
                ext = parts[1].lower()
        
        icon = self.ICONS.get(ext, self.ICONS["default"])
        self.icon_label.setText(icon)
    
    def clear_content(self):
        """Clear for recycling."""
        self.icon_label.clear()
        self.title_label.clear()
        self.subtitle_label.clear()
