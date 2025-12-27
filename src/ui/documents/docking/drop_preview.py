"""
Drop Preview

Dotted outline showing the result of a drop operation.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen
from .drop_zone_overlay import DropZone


class DropPreview(QWidget):
    """
    Dotted outline preview of drop result.
    
    Shows semi-transparent preview of where document
    will appear after drop operation.
    """
    
    def __init__(self, zone: DropZone, parent: Optional[QWidget] = None):
        """
        Initialize drop preview.
        
        Args:
            zone: Drop zone this preview represents
            parent: Parent widget
        """
        super().__init__(parent)
        self.zone = zone
        
        # Window configuration
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
    
    def paintEvent(self, event):
        """
        Draw dotted preview rectangle.
        
        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dotted blue border
        pen = QPen(QColor("#0078D4"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # Semi-transparent fill
        painter.fillRect(self.rect(), QColor(0, 120, 212, 15))  # 6% opacity
