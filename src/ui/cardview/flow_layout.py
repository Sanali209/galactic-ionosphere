"""
FlowLayout - Responsive grid layout for card items.

Ported from pySide6Ext with improvements for Foundation.
Arranges widgets in a flowing grid that wraps automatically.
"""
from PySide6.QtCore import QRect, QSize, QPoint, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget, QSizePolicy


class FlowLayout(QLayout):
    """
    Flow layout that arranges widgets in rows, wrapping to next row when full.
    
    Features:
    - Responsive grid that adapts to container width
    - Configurable horizontal and vertical spacing
    - Works with QScrollArea for virtualization
    - Efficient height-for-width calculation
    
    Example:
        layout = FlowLayout(margin=8, h_spacing=8, v_spacing=8)
        for widget in widgets:
            layout.addWidget(widget)
    """
    
    def __init__(
        self, 
        parent: QWidget | None = None, 
        margin: int = 0, 
        h_spacing: int = 8, 
        v_spacing: int = 8
    ):
        """
        Initialize FlowLayout.
        
        Args:
            parent: Parent widget
            margin: Margin around content
            h_spacing: Horizontal spacing between items
            v_spacing: Vertical spacing between rows
        """
        super().__init__(parent)
        self._item_list: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)
    
    @property
    def h_spacing(self) -> int:
        """Horizontal spacing between items."""
        return self._h_spacing
    
    @h_spacing.setter
    def h_spacing(self, value: int):
        self._h_spacing = value
        self.invalidate()
    
    @property
    def v_spacing(self) -> int:
        """Vertical spacing between rows."""
        return self._v_spacing
    
    @v_spacing.setter
    def v_spacing(self, value: int):
        self._v_spacing = value
        self.invalidate()
    
    def addItem(self, item: QLayoutItem):
        """Add item to layout."""
        self._item_list.append(item)
    
    def count(self) -> int:
        """Return number of items."""
        return len(self._item_list)
    
    def itemAt(self, index: int) -> QLayoutItem | None:
        """Get item at index."""
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None
    
    def takeAt(self, index: int) -> QLayoutItem | None:
        """Remove and return item at index."""
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None
    
    def expandingDirections(self) -> Qt.Orientation:
        """Return expanding directions."""
        return Qt.Orientation(0)
    
    def hasHeightForWidth(self) -> bool:
        """This layout's height depends on width."""
        return True
    
    def heightForWidth(self, width: int) -> int:
        """Calculate height needed for given width."""
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)
    
    def setGeometry(self, rect: QRect):
        """Set layout geometry."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return self.minimumSize()
    
    def minimumSize(self) -> QSize:
        """Return minimum size."""
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom()
        )
        return size
    
    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        """
        Perform layout calculation.
        
        Args:
            rect: Available rectangle
            test_only: If True, only calculate without moving widgets
            
        Returns:
            Total height used
        """
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(
            margins.left(), margins.top(),
            -margins.right(), -margins.bottom()
        )
        
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        
        for item in self._item_list:
            widget = item.widget()
            if widget and not widget.isVisible():
                continue
                
            item_size = item.sizeHint()
            next_x = x + item_size.width() + self._h_spacing
            
            # Wrap to next line if needed
            if next_x - self._h_spacing > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + self._v_spacing
                next_x = x + item_size.width() + self._h_spacing
                line_height = 0
            
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            
            x = next_x
            line_height = max(line_height, item_size.height())
        
        return y + line_height - rect.y() + margins.bottom()
    
    def clear(self):
        """Remove all items from layout."""
        while self.count():
            item = self.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
    
    def get_items_per_row(self, item_width: int) -> int:
        """
        Calculate how many items fit per row.
        
        Args:
            item_width: Width of each item
            
        Returns:
            Number of items per row
        """
        margins = self.contentsMargins()
        available_width = self.geometry().width() - margins.left() - margins.right()
        if available_width <= 0:
            return 1
        
        items = max(1, (available_width + self._h_spacing) // (item_width + self._h_spacing))
        return items
    
    def update_layout(self):
        """Force recalculation of all widget positions."""
        rect = self.geometry()
        if rect.width() > 0 and rect.height() > 0:
            self._do_layout(rect, test_only=False)
