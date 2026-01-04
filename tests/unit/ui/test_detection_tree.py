import pytest
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Ensure path
sys.path.append(os.path.join(os.getcwd(), 'samples', 'uexplorer'))

from uexplorer_src.ui.widgets.detection_tree import DetectionTreeWidget

app = QApplication.instance() or QApplication([])

def test_tree_loading():
    widget = DetectionTreeWidget()
    
    data = [
        {'class_name': 'Person', 'group_name': 'face', 'count': 10},
        {'class_name': 'Person', 'group_name': 'body', 'count': 5},
        {'class_name': 'Car', 'group_name': 'any', 'count': 3}
    ]
    
    widget.load_data(data)
    
    # Root items: Person, Car
    assert widget.topLevelItemCount() == 2
    
    # Person
    person = widget.findItems("Person", Qt.MatchFlag.MatchExactly)[0]
    assert person.text(1) == "15"  # Total count
    assert person.childCount() == 2 # face, body
    
    # Car
    car = widget.findItems("Car", Qt.MatchFlag.MatchExactly)[0]
    assert car.text(1) == "3"
    assert car.childCount() == 0 # 'any' group usually merged or hidden if only one?
    # Logic in implementation: "if len(groups) > 1 or (len(groups) == 1 and 'any' not in groups)"
    # Car has 'any' and len=1 -> no children. Correct.

def test_interaction_signals(qtbot):
    """Test signaling using qtbot if available, or manual."""
    widget = DetectionTreeWidget()
    data = [{'class_name': 'Person', 'group_name': 'any', 'count': 10}]
    widget.load_data(data)
    
    # Mock signal
    filters_emitted = []
    widget.filter_changed.connect(lambda f: filters_emitted.append(f))
    
    # Toggle Person
    person = widget.topLevelItem(0)
    person.setCheckState(0, Qt.CheckState.Checked)
    
    # Signal should be emitted
    assert len(filters_emitted) > 0
    assert filters_emitted[0][0]['class_name'] == 'Person'
    assert filters_emitted[0][0]['negate'] is False

def test_set_active_filters():
    widget = DetectionTreeWidget()
    data = [{'class_name': 'A', 'count': 1}, {'class_name': 'B', 'count': 1}]
    widget.load_data(data)
    
    filters = [{'class_name': 'A', 'group_name': 'any', 'min_count': 1, 'negate': False}]
    widget.set_active_filters(filters)
    
    item_a = widget.findItems("A", Qt.MatchFlag.MatchExactly)[0]
    assert item_a.checkState(0) == Qt.CheckState.Checked
    
    item_b = widget.findItems("B", Qt.MatchFlag.MatchExactly)[0]
    assert item_b.checkState(0) == Qt.CheckState.Unchecked
