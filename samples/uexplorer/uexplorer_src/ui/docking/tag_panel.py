"""
Dockable Tag Panel for UExplorer.

Works with DockingService (QWidget-based).
Supports include/exclude filtering.
"""
from typing import TYPE_CHECKING, List, Set, Optional
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
from PySide6.QtCore import Signal

import sys
from pathlib import Path
from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.tag_tree import TagTreeWidget

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator

class TagPanel(PanelBase):
    """
    Tag browser panel.
    
    Shows hierarchical tag structure and allows:
    - Browsing tags
    - Filtering by tags (include/exclude)
    - Drag & drop files onto tags
    
    Signals:
        filter_changed(include_ids, exclude_ids): Emitted when tag filter changes
    """
    
    # Filter changed signal for unified search
    filter_changed = Signal(list, list)  # (include_tag_ids, exclude_tag_ids)
    
    def __init__(self, parent: Optional[QWidget], locator: "ServiceLocator") -> None:
        """
        Args:
            parent: Parent widget
            locator: ServiceLocator for services
        """
        self._tree: Optional[TagTreeWidget] = None
        self._include_tags: Set[str] = set()
        self._exclude_tags: Set[str] = set()
        super().__init__(locator, parent)
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with filter controls
        header = QHBoxLayout()
        
        title = QLabel("Tags")
        title.setStyleSheet("font-weight: bold; color: #ffffff;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Clear filters button
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(20)
        self.btn_clear.setToolTip("Clear all tag filters")
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #5a5a5a;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #6a6a6a; }
        """)
        self.btn_clear.clicked.connect(self._clear_filters)
        self.btn_clear.hide()  # Show only when filters active
        header.addWidget(self.btn_clear)
        
        layout.addLayout(header)
        
        # Filter status label
        self.filter_label = QLabel("")
        self.filter_label.setStyleSheet("color: #888; font-size: 11px;")
        self.filter_label.hide()
        layout.addWidget(self.filter_label)
        
        # Tag tree
        self._tree = TagTreeWidget(self.locator)
        layout.addWidget(self._tree)
        
        # Connect include/exclude signals
        self._tree.include_requested.connect(self.toggle_include)
        self._tree.exclude_requested.connect(self.toggle_exclude)
        
        # Connect single-click to replace filter
        if hasattr(self._tree, 'itemClicked'):
            self._tree.itemClicked.connect(self._on_tag_clicked)
        
        # Hint label
        hint = QLabel("Click to filter | Right-click: Add/Exclude")
        hint.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(hint)
    
    @property
    def tree(self) -> TagTreeWidget:
        return self._tree
    
    @property
    def include_ids(self) -> List[str]:
        """Get list of included tag IDs."""
        return list(self._include_tags)
    
    @property
    def exclude_ids(self) -> List[str]:
        """Get list of excluded tag IDs."""
        return list(self._exclude_tags)
    
    def _on_tag_clicked(self, item, column):
        """Handle tag item clicked - replace filter section with this tag."""
        if not item:
            return
        
        # Get tag ID from item
        tag_id = item.data(0, 0x0100)  # Qt.UserRole
        if not tag_id:
            return
        
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        # Check if Ctrl is pressed - if so, add to filter instead of replacing
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Add to filter without replacing
            self.toggle_include(tag_id)
        else:
            # Replace entire tag filter section with this tag
            self.replace_tag_filter(tag_id)
    
    def replace_tag_filter(self, tag_id: str):
        """Replace all tag filters with just this tag."""
        # Clear existing tag filters
        self._include_tags.clear()
        self._exclude_tags.clear()
        
        # Set only this tag as included
        self._include_tags.add(tag_id)
        
        # Emit change
        self._emit_filter_changed()
    
    def toggle_include(self, tag_id: str):
        """Toggle tag in include list."""
        if tag_id in self._include_tags:
            self._include_tags.discard(tag_id)
        else:
            self._include_tags.add(tag_id)
            # Remove from exclude if present
            self._exclude_tags.discard(tag_id)
        self._emit_filter_changed()
    
    def toggle_exclude(self, tag_id: str):
        """Toggle tag in exclude list."""
        if tag_id in self._exclude_tags:
            self._exclude_tags.discard(tag_id)
        else:
            self._exclude_tags.add(tag_id)
            # Remove from include if present
            self._include_tags.discard(tag_id)
        self._emit_filter_changed()
    
    def _clear_filters(self):
        """Clear all tag filters."""
        self._include_tags.clear()
        self._exclude_tags.clear()
        self._emit_filter_changed()
    
    def _update_ui(self):
        """Update the filter status label and clear button visibility."""
        include = list(self._include_tags)
        exclude = list(self._exclude_tags)
        
        if include or exclude:
            parts = []
            if include:
                parts.append(f"+{len(include)}")
            if exclude:
                parts.append(f"-{len(exclude)}")
            self.filter_label.setText(f"Filter: {' '.join(parts)}")
            self.filter_label.show()
            self.btn_clear.show()
        else:
            self.filter_label.hide()
            self.btn_clear.hide()

    def _emit_filter_changed(self):
        """Update UI and emit filter changed signal."""
        self._update_ui()
        include = list(self._include_tags)
        exclude = list(self._exclude_tags)
        self.filter_changed.emit(include, exclude)
    
    def on_update(self, context=None):
        """Refresh tags when panel updated."""
        if self._tree:
            import asyncio
            asyncio.ensure_future(self._tree.refresh_tags())
    
    def get_state(self) -> dict:
        """Save panel state for persistence."""
        state = {}
        if self._tree and hasattr(self._tree, 'model'):
            try:
                expanded_items = self._tree.get_expanded_items() if hasattr(self._tree, 'get_expanded_items') else []
                if expanded_items:
                    state['expanded_tags'] = expanded_items
            except:
                pass
        
        # Save filter state
        if self._include_tags:
            state['include_tags'] = list(self._include_tags)
        if self._exclude_tags:
            state['exclude_tags'] = list(self._exclude_tags)
        
        return state
    
    def set_state(self, state: dict):
        """Restore panel state from saved data."""
        if not state or not self._tree:
            return
        
        if 'expanded_tags' in state and hasattr(self._tree, 'expand_items'):
            try:
                self._tree.expand_items(state['expanded_tags'])
            except:
                pass
        
        # Restore filter state
        if 'include_tags' in state:
            self._include_tags = set(state['include_tags'])
        if 'exclude_tags' in state:
            self._exclude_tags = set(state['exclude_tags'])
        
        if self._include_tags or self._exclude_tags:
            self._emit_filter_changed()
    
    def clear_filters(self):
        """Clear all include/exclude filters."""
        self._include_tags.clear()
        self._exclude_tags.clear()
        self._update_ui()
        # Don't emit - already handled by UnifiedQueryBuilder.clear_all()
