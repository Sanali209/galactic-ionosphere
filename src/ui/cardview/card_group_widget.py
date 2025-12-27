"""
CardGroupWidget - Collapsible group header widget.

Provides grouping functionality similar to Photoshop Elements:
- Click to expand/collapse
- Group count badge
- FlowLayout for items
"""
from typing import List
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSizePolicy, QFrame
)
from loguru import logger

from src.ui.cardview.flow_layout import FlowLayout
from src.ui.cardview.card_item_widget import CardItemWidget


class CardGroupWidget(QWidget):
    """
    Collapsible group container for card items.
    
    Features:
    - Clickable header to expand/collapse
    - Item count badge
    - FlowLayout for item arrangement
    - Customizable header styling
    
    Signals:
        collapsed_changed(group_key: str, is_collapsed: bool)
    
    Example:
        group = CardGroupWidget("Photos")
        group.add_item(card_widget)
        group.set_collapsed(True)
    """
    
    collapsed_changed = Signal(str, bool)
    
    def __init__(self, group_key: str, parent: QWidget | None = None):
        """
        Initialize group widget.
        
        Args:
            group_key: Unique key/name for this group
            parent: Parent widget
        """
        super().__init__(parent)
        self.group_key = group_key
        self._collapsed = False
        self._item_widgets: List[CardItemWidget] = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)
        
        # --- Header ---
        self.header = QFrame()
        self.header.setObjectName("groupHeader")
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.mousePressEvent = self._on_header_click
        
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)
        
        # Collapse indicator
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setObjectName("collapseBtn")
        self.collapse_btn.setFixedSize(24, 24)
        self.collapse_btn.clicked.connect(self.toggle_collapsed)
        self.collapse_btn.setStyleSheet("""
            QPushButton#collapseBtn {
                background: transparent;
                border: none;
                font-size: 12px;
                color: #666;
            }
            QPushButton#collapseBtn:hover {
                color: #333;
            }
        """)
        
        # Group title
        self.title_label = QLabel(self.group_key)
        self.title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 13px;
                color: #333;
            }
        """)
        
        # Item count
        self.count_label = QLabel("(0)")
        self.count_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 12px;
            }
        """)
        
        header_layout.addWidget(self.collapse_btn)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()
        
        # Header styling
        self.header.setStyleSheet("""
            QFrame#groupHeader {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QFrame#groupHeader:hover {
                background-color: #ebebeb;
            }
        """)
        
        layout.addWidget(self.header)
        
        # --- Content (FlowLayout for items) ---
        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Minimum
        )
        self.items_layout = FlowLayout(
            self.content_widget,
            margin=4,
            h_spacing=8,
            v_spacing=8
        )
        
        layout.addWidget(self.content_widget)
    
    # --- Collapse/Expand ---
    
    @property
    def is_collapsed(self) -> bool:
        """Check if group is collapsed."""
        return self._collapsed
    
    def toggle_collapsed(self):
        """Toggle between collapsed and expanded."""
        self.set_collapsed(not self._collapsed)
    
    def set_collapsed(self, collapsed: bool):
        """
        Set collapsed state.
        
        Args:
            collapsed: True to collapse, False to expand
        """
        if self._collapsed == collapsed:
            return
        
        self._collapsed = collapsed
        self.collapse_btn.setText("▶" if collapsed else "▼")
        
        if collapsed:
            # Hide content
            self.content_widget.hide()
        else:
            # Show content first
            self.content_widget.show()
            
            # Force re-layout all widgets
            for widget in self._item_widgets:
                widget.show()
                widget.raise_()
            
            # Force layout to recalculate all positions
            self.items_layout.update_layout()
            self.content_widget.updateGeometry()
            self.updateGeometry()
        
        self.collapsed_changed.emit(self.group_key, collapsed)
        
        logger.debug(f"Group '{self.group_key}' {'collapsed' if collapsed else 'expanded'}, widgets: {len(self._item_widgets)}")
    
    def _on_header_click(self, event):
        """Handle header click to toggle."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_collapsed()
    
    # --- Item Management ---
    
    def add_item(self, widget: CardItemWidget):
        """
        Add item widget to group.
        
        Args:
            widget: Card widget to add
        """
        self._item_widgets.append(widget)
        self.items_layout.addWidget(widget)
        self._update_count()
    
    def remove_item(self, widget: CardItemWidget):
        """
        Remove item widget from group.
        
        Args:
            widget: Card widget to remove
        """
        if widget in self._item_widgets:
            self._item_widgets.remove(widget)
            self.items_layout.removeWidget(widget)
            self._update_count()
    
    def clear_items(self):
        """Remove all items from group."""
        for widget in self._item_widgets[:]:
            self.remove_item(widget)
        self._item_widgets.clear()
        self._update_count()
    
    @property
    def item_count(self) -> int:
        """Get number of items in group."""
        return len(self._item_widgets)
    
    @property
    def item_widgets(self) -> List[CardItemWidget]:
        """Get list of item widgets."""
        return self._item_widgets.copy()
    
    def _update_count(self):
        """Update count label."""
        self.count_label.setText(f"({self.item_count})")
    
    # --- Customization ---
    
    def set_title(self, title: str):
        """Set group title."""
        self.title_label.setText(title)
    
    def set_header_color(self, color: str):
        """Set header background color."""
        self.header.setStyleSheet(f"""
            QFrame#groupHeader {{
                background-color: {color};
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
        """)
