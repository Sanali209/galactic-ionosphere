"""
Filter Badge Widget - Removable chip for active filters.

Displays a single filter with visual indicators and remove button.
Style inspired by Material Design 3 chips.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class FilterBadge(QWidget):
    """
    Single filter badge/chip with remove button.
    
    Shows: [+/-] FilterName [X]
    
    Signals:
        removed: Emitted when X button is clicked
    """
    
    removed = Signal()
    
    def __init__(self, text: str, filter_type: str, include: bool = True, parent: Optional[QWidget] = None) -> None:
        """
        Create a filter badge.
        
        Args:
            text: Display text for the filter (tag name, path, etc.)
            filter_type: "tag", "directory", "album"
            include: True for include (+), False for exclude (-)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.text = text
        self.filter_type = filter_type
        self.include = include
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Build the badge UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        
        # Include/Exclude indicator
        self.indicator = QLabel("+" if self.include else "-")
        indicator_font = QFont()
        indicator_font.setBold(True)
        indicator_font.setPointSize(11)
        self.indicator.setFont(indicator_font)
        layout.addWidget(self.indicator)
        
        # Filter text
        self.label = QLabel(self._truncate_text(self.text, 30))
        self.label.setToolTip(self.text)  # Full text on hover
        layout.addWidget(self.label)
        
        # Remove button
        self.btn_remove = QPushButton("×")
        self.btn_remove.setFixedSize(16, 16)
        self.btn_remove.setToolTip("Remove filter")
        self.btn_remove.clicked.connect(self.removed.emit)
        layout.addWidget(self.btn_remove)
        
        # Make widget compact
        self.setMaximumHeight(28)
    
    def _truncate_text(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[:max_len-1] + "…"
    
    def _apply_style(self):
        """Apply styling based on filter type and include/exclude."""
        # Color scheme per type - improved contrast
        colors = {
            "tag": {
                "include": {"bg": "#3a6ea5", "fg": "#ffffff", "indicator": "#6ec8ff"},
                "exclude": {"bg": "#a53a3a", "fg": "#ffffff", "indicator": "#ff6e6e"}
            },
            "directory": {
                "include": {"bg": "#4a8a3a", "fg": "#ffffff", "indicator": "#8aff6e"},
                "exclude": {"bg": "#a58a3a", "fg": "#ffffff", "indicator": "#ffc86e"}
            },
            "album": {
                "include": {"bg": "#7a3aa5", "fg": "#ffffff", "indicator": "#d88aff"},
                "exclude": {"bg": "#a53a7a", "fg": "#ffffff", "indicator": "#ff8ad8"}
            },
            "detection": {
                "include": {"bg": "#20b2aa", "fg": "#ffffff", "indicator": "#e0ffff"},
                "exclude": {"bg": "#cd5c5c", "fg": "#ffffff", "indicator": "#ffb6c1"}
            }
        }
        
        mode = "include" if self.include else "exclude"
        color_set = colors.get(self.filter_type, colors["tag"])[mode]
        
        self.setStyleSheet(f"""
            FilterBadge {{
                background-color: {color_set["bg"]};
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            FilterBadge:hover {{
                background-color: {color_set["bg"]}ee;
                border: 1px solid rgba(255, 255, 255, 0.4);
            }}
        """)
        
        # Ensure text is readable with bold font
        label_font = self.label.font()
        label_font.setBold(False)
        self.label.setFont(label_font)
        self.label.setStyleSheet(f"color: {color_set['fg']}; background: transparent; font-weight: 500;")
        
        # Make indicator more visible
        self.indicator.setStyleSheet(f"color: {color_set['indicator']}; background: transparent; font-weight: bold;")
        
        self.btn_remove.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 255, 255, 0.15);
                color: {color_set['fg']};
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.3);
            }}
            QPushButton:pressed {{
                background-color: rgba(255, 255, 255, 0.4);
            }}
        """)
