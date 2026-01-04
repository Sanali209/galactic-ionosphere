import pytest
from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QApplication

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'samples', 'uexplorer'))

from uexplorer_src.ui.widgets.detection_graphics_items import DetectionRectItem

# Needed for QGraphicsItem
app = QApplication.instance() or QApplication([])

def test_detection_rect_init():
    item = DetectionRectItem(10, 20, 100, 200, label="Test", score=0.9)
    assert item.rect() == QRectF(10, 20, 100, 200)
    assert item._label == "Test"
    assert item._score == 0.9

def test_handles_init():
    item = DetectionRectItem(0, 0, 100, 100)
    # Accessible via private member for testing
    assert len(item._handles) == 4
    
    # Handles should be hidden by default
    assert not item._handles[0].isVisible()

def test_editable_handles_visibility():
    item = DetectionRectItem(0, 0, 100, 100)
    item.setSelected(True)
    
    # Initially not editable -> handles hidden even if selected
    item.set_editable(False)
    assert not item._handles[0].isVisible()
    
    # Editable -> handles shown if selected
    item.set_editable(True)
    assert item._handles[0].isVisible()
    
    # Deselect -> handles hidden
    item.setSelected(False)
    # The itemChange event handles visibility update
    # In headless test without Scene, itemChange might not trigger automatically 
    # unless we simulate selection change properly or call the handler.
    # QGraphicsItem.setSelected DOES trigger itemChange.
    
    assert not item._handles[0].isVisible()

def test_resize_logic():
    # This involves callbacks.
    item = DetectionRectItem(0, 0, 100, 100)
    
    # Simulate handle move
    # Handle 0 is TL (Top Left)
    handle_tl = item._handles[0]
    
    # The callback is inside _init_handles closure but assigned to handle
    # We can simulate the callback execution logic by calling setRect directly
    # or by invoking the callback stored in handle if we exposed it.
    # But `_on_move` is private/protected on handle instance.
    
    # We'll just verify setRect logic which is what the handle calls.
    item.setRect(QRectF(10, 10, 90, 90))
    assert item.rect() == QRectF(10, 10, 90, 90)
    
    # Verify handles updated positions
    # TL handle should be at 10, 10
    assert handle_tl.pos() == QPointF(10, 10)
