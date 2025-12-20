"""
Drop Zone Overlay

Visual overlay showing 5 drop zones during document drag operations.
Provides compass-style interface matching Visual Studio behavior.
"""
from typing import Optional, List, Dict
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent
from enum import Enum
from loguru import logger


class DropZone(Enum):
    """Drop zone locations."""
    CENTER = 0    # Add to existing tab group
    LEFT = 1      # Create left split
    RIGHT = 2     # Create right split
    TOP = 3       # Create top split
    BOTTOM = 4    # Create bottom split
    NONE = -1     # No zone


class DropZoneOverlay(QWidget):
    """
    Semi-transparent overlay showing drop zones.
    
    Displays compass-style interface with 5 zones:
    - CENTER: Add to existing tabs
    - LEFT/RIGHT/TOP/BOTTOM: Create new split
    
    Signals:
        zone_entered: Emitted when cursor enters a zone
        zone_exited: Emitted when cursor leaves a zone
        drop_requested: Emitted when drop is requested
    """
    
    zone_entered = Signal(DropZone)
    zone_exited = Signal()
    drop_requested = Signal(DropZone)
    
    def __init__(self, parent_widget: QWidget, show_split_zones: bool = True):
        """
        Initialize drop zone overlay.
        
        Args:
            parent_widget: Widget to overlay (typically a SplitContainer) - MUST be parent!
            show_split_zones: If True, show all 5 zones. If False, only show CENTER for tab reordering
        """
        super().__init__(parent_widget)  # CRITICAL: Must be CHILD of container to receive events!
        self.parent_widget = parent_widget
        self.show_split_zones = show_split_zones
        self.zones: List[Dict] = []
        self.hover_zone: Optional[DropZone] = None
        
        # Widget configuration - child widget that covers parent
        # NO window flags - we're a regular child widget!
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # MUST receive events!
        self.setMouseTracking(True)
        self.setAcceptDrops(True)  # CRITICAL for receiving drop events
        
        # Position to cover entire parent
        self.setGeometry(parent_widget.rect())
        
        # Raise to top of stacking order
        self.raise_()
        
        logger.debug("DropZoneOverlay created as child widget")
    
    def showEvent(self, event):
        """Recalculate zones when shown."""
        self._calculate_zones()
        logger.debug("DropZoneOverlay shown, zones calculated")
        super().showEvent(event)
    
    def _calculate_zones(self):
        """Calculate drop zone rectangles based on parent widget size."""
        w = self.width()
        h = self.height()
        
        # Define zone sizes
        edge_size = min(w, h) // 3  # Edge zones are 1/3 of smallest dimension
        center_margin = edge_size // 2  # Center zone margin
        
        self.zones = []
        
        if self.show_split_zones:
            # Show all 5 zones for splitting
            # TOP zone
            self.zones.append({
                'zone': DropZone.TOP,
                'rect': QRect(edge_size, 0, w - 2 * edge_size, edge_size),
                'icon_pos': QPoint(w // 2, edge_size // 2)
            })
            
            # BOTTOM zone
            self.zones.append({
                'zone': DropZone.BOTTOM,
                'rect': QRect(edge_size, h - edge_size, w - 2 * edge_size, edge_size),
                'icon_pos': QPoint(w // 2, h - edge_size // 2)
            })
            
            # LEFT zone
            self.zones.append({
                'zone': DropZone.LEFT,
                'rect': QRect(0, edge_size, edge_size, h - 2 * edge_size),
                'icon_pos': QPoint(edge_size // 2, h // 2)
            })
            
            # RIGHT zone
            self.zones.append({
                'zone': DropZone.RIGHT,
                'rect': QRect(w - edge_size, edge_size, edge_size, h - 2 * edge_size),
                'icon_pos': QPoint(w - edge_size // 2, h // 2)
            })
        
        # CENTER zone (always shown)
        self.zones.append({
            'zone': DropZone.CENTER,
            'rect': QRect(center_margin, center_margin, w - 2 * center_margin, h - 2 * center_margin),
            'icon_pos': QPoint(w // 2, h // 2)
        })
        
        logger.debug(f"Zones calculated: {len(self.zones)} zones")
    
    def dragEnterEvent(self, event):
        """Accept drag when it enters overlay."""
        if event.mimeData().hasFormat("application/x-foundation-document"):
            event.acceptProposedAction()
            logger.debug("Drag entered overlay")
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """
        Detect which zone cursor is over during drag.
        
        Args:
            event: Drag move event
        """
        if not event.mimeData().hasFormat("application/x-foundation-document"):
            event.ignore()
            return
        
        zone = self._hit_test(event.pos())
        
        if zone != self.hover_zone:
            # Zone changed
            if self.hover_zone and self.hover_zone != DropZone.NONE:
                self.zone_exited.emit()
                logger.debug(f"Exited zone: {self.hover_zone.name}")
            
            self.hover_zone = zone
            
            if zone != DropZone.NONE:
                self.zone_entered.emit(zone)
                logger.debug(f"Entered zone: {zone.name}")
            
            self.update()  # Repaint
        
        event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        """Clear zone highlight when drag leaves."""
        if self.hover_zone and self.hover_zone != DropZone.NONE:
            self.zone_exited.emit()
            logger.debug(f"Drag left overlay, was in: {self.hover_zone.name}")
        
        self.hover_zone = None
        self.update()
    
    def dropEvent(self, event):
        """
        Handle drop request.
        
        Args:
            event: Drop event
        """
        if not event.mimeData().hasFormat("application/x-foundation-document"):
            event.ignore()
            return
        
        # Get the zone where drop occurred
        zone = self._hit_test(event.pos())
        
        if zone and zone != DropZone.NONE:
            logger.info(f"Drop accepted in zone: {zone.name}")
            # Emit signal so drag coordinator can handle it
            self.drop_requested.emit(zone)
            # IMPORTANT: Accept the drop action so Qt completes it
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            logger.warning("Drop outside any zone")
            event.ignore()
    

    
    def _hit_test(self, pos: QPoint) -> DropZone:
        """
        Determine which zone contains point.
        
        Args:
            pos: Point to test
            
        Returns:
            DropZone containing the point, or NONE
        """
        for zone_info in self.zones:
            if zone_info['rect'].contains(pos):
                return zone_info['zone']
        
        return DropZone.NONE
    
    def paintEvent(self, event):
        """
        Draw zone highlights.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw ALL zones with subtle outline so user can see them
        painter.setPen(QPen(QColor(0, 120, 212, 100), 1))  # Subtle blue outline
        for zone_info in self.zones:
            rect = zone_info['rect']
            painter.drawRect(rect)
        
        # Highlight current zone with bright fill and thick border
        if self.hover_zone and self.hover_zone != DropZone.NONE:
            # Find the rect for this zone
            for zone_info in self.zones:
                if zone_info['zone'] == self.hover_zone:
                    rect = zone_info['rect']
                    
                    # Bright fill based on zone type
                    if self.hover_zone == DropZone.CENTER:
                        fill_color = QColor(0, 180, 0, 100)  # Green for center (reorder)
                    else:
                        fill_color = QColor(0, 120, 212, 120)  # Blue for edges (split)
                    
                    painter.fillRect(rect, fill_color)
                    
                    # Thick bright border
                    painter.setPen(QPen(QColor(255, 255, 255, 200), 4))  # Thick white border
                    painter.drawRect(rect.adjusted(2, 2, -2, -2))
                    
                    # Inner accent border
                    if self.hover_zone == DropZone.CENTER:
                        painter.setPen(QPen(QColor(0, 255, 0), 2))  # Bright green
                    else:
                        painter.setPen(QPen(QColor(0, 180, 255), 2))  # Bright blue
                    painter.drawRect(rect.adjusted(4, 4, -4, -4))
                    
                    logger.debug(f"Painted zone: {self.hover_zone.name}")
                    break
