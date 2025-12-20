"""
Tests for Drop Zone Overlay System

Validates zone detection, boundaries, and visual feedback.
"""
import pytest
from PySide6.QtCore import QPoint, QRect, Qt, QEvent
from PySide6.QtGui import QMouseEvent
from src.ui.documents.docking.drop_zone_overlay import DropZone, DropZoneOverlay
from src.ui.documents.docking.drop_preview import DropPreview


class TestDropZoneOverlay:
    """Test drop zone hit detection and visual feedback."""
    
    def setup_method(self):
        """Create overlay for testing."""
        self.overlay = DropZoneOverlay()
        self.overlay.resize(800, 600)
        self.overlay._calculate_zones()
    
    def test_center_zone_detection(self):
        """Test center zone covers middle area."""
        center_point = QPoint(400, 300)  # Dead center
        zone = self.overlay._hit_test(center_point)
        assert zone == DropZone.CENTER
    
    def test_left_edge_zone(self):
        """Test left edge zone (first 20%)."""
        left_point = QPoint(80, 300)  # ~10% from left
        zone = self.overlay._hit_test(left_point)
        assert zone == DropZone.LEFT
    
    def test_right_edge_zone(self):
        """Test right edge zone (last 20%)."""
        right_point = QPoint(720, 300)  # ~90% from left
        zone = self.overlay._hit_test(right_point)
        assert zone == DropZone.RIGHT
    
    def test_top_edge_zone(self):
        """Test top edge zone (first 20% height)."""
        top_point = QPoint(400, 60)  # ~10% from top
        zone = self.overlay._hit_test(top_point)
        assert zone == DropZone.TOP
    
    def test_bottom_edge_zone(self):
        """Test bottom edge zone (last 20% height)."""
        bottom_point = QPoint(400, 540)  # ~90% from top
        zone = self.overlay._hit_test(bottom_point)
        assert zone == DropZone.BOTTOM
    
    def test_corner_priority(self):
        """Test corners prioritize edges over center."""
        top_left = QPoint(80, 60)
        zone = self.overlay._hit_test(top_left)
        # Should hit TOP before LEFT (order matters)
        assert zone in [DropZone.TOP, DropZone.LEFT]
    
    def test_zone_boundary_calculation(self):
        """Verify zone rectangles are calculated correctly."""
        assert DropZone.LEFT in self.overlay._zone_rects
        assert DropZone.CENTER in self.overlay._zone_rects
        
        left_rect = self.overlay._zone_rects[DropZone.LEFT]
        assert left_rect.x() == 0
        assert left_rect.width() == 160  # 20% of 800
    
    def test_zone_entered_signal(self, qtbot):
        """Test signal emission when zone changes."""
        with qtbot.waitSignal(self.overlay.zone_entered, timeout=1000) as blocker:
            # Simulate mouse move to LEFT zone
            event = QMouseEvent(
                QEvent.MouseMove,
                QPoint(80, 300),
                Qt.NoButton,
                Qt.NoButton,
                Qt.NoModifier
            )
            self.overlay.mouseMoveEvent(event)
        
        assert blocker.args[0] == DropZone.LEFT
    
    def test_zone_exited_signal(self, qtbot):
        """Test signal emission when leaving zone."""
        # First enter a zone
        self.overlay._current_zone = DropZone.LEFT
        
        with qtbot.waitSignal(self.overlay.zone_exited, timeout=1000) as blocker:
            # Move to different zone
            event = QMouseEvent(
                QEvent.MouseMove,
                QPoint(400, 300),  # CENTER
                Qt.NoButton,
                Qt.NoButton,
                Qt.NoModifier
            )
            self.overlay.mouseMoveEvent(event)
        
        # Signal should have been emitted
        assert blocker.signal_triggered
    
    def test_drop_requested_signal(self, qtbot):
        """Test drop signal on mouse release."""
        # Set current zone
        self.overlay._current_zone = DropZone.CENTER
        
        with qtbot.waitSignal(self.overlay.drop_requested, timeout=1000) as blocker:
            event = QMouseEvent(
                QEvent.MouseButtonRelease,
                QPoint(400, 300),
                Qt.LeftButton,
                Qt.NoButton,
                Qt.NoModifier
            )
            self.overlay.mouseReleaseEvent(event)
        
        assert blocker.args[0] == DropZone.CENTER


class TestDropPreview:
    """Test drop preview rendering."""
    
    def test_preview_creation(self):
        """Verify preview can be created for each zone."""
        for zone in [DropZone.LEFT, DropZone.RIGHT, DropZone.TOP, DropZone.BOTTOM, DropZone.CENTER]:
            preview = DropPreview(zone)
            assert preview.zone == zone
    
    def test_preview_window_flags(self):
        """Verify preview has correct window flags."""
        preview = DropPreview(DropZone.CENTER)
        flags = preview.windowFlags()
        
        assert flags & Qt.FramelessWindowHint
        assert flags & Qt.Tool
    
    def test_preview_attributes(self):
        """Verify preview has transparent background."""
        preview = DropPreview(DropZone.LEFT)
        assert preview.testAttribute(Qt.WA_TranslucentBackground)


class TestDropZoneEnum:
    """Test DropZone enum."""
    
    def test_enum_values(self):
        """Verify all zone values exist."""
        assert DropZone.CENTER.value == 0
        assert DropZone.LEFT.value == 1
        assert DropZone.RIGHT.value == 2
        assert DropZone.TOP.value == 3
        assert DropZone.BOTTOM.value == 4
        assert DropZone.NONE.value == -1
