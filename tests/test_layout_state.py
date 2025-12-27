"""
Unit tests for layout state serialization (Phase 1).
"""
import pytest
import json
import tempfile
from pathlib import Path
from src.ui.documents.split_manager import SplitManager, SplitOrientation
from src.ui.documents.layout_state import LayoutState

def test_split_tree_serialization():
    """Test converting split tree to JSON dict."""
    manager = SplitManager()
    manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    data = manager.to_dict()
    
    assert "root" in data
    assert "active_node_id" in data
    assert data["root"]["is_container"] == False
    assert data["root"]["orientation"] == "HORIZONTAL"
    assert len(data["root"]["children"]) == 2

def test_split_tree_deserialization():
    """Test recreating split tree from JSON dict."""
    # Create original
    manager1 = SplitManager()
    id1 = manager1.split_node(manager1.root.id, SplitOrientation.VERTICAL)
    
    # Serialize
    data = manager1.to_dict()
    
    # Deserialize
    manager2 = SplitManager.from_dict(data)
    
    assert not manager2.root.is_container
    assert manager2.root.orientation == SplitOrientation.VERTICAL
    assert len(manager2.root.children) == 2

def test_round_trip_serialization():
    """Test serialize → deserialize → compare."""
    manager1 = SplitManager()
    manager1.split_node(manager1.root.id, SplitOrientation.HORIZONTAL)
    first_child = manager1.root.children[0]
    manager1.split_node(first_child.id, SplitOrientation.VERTICAL)
    
    # Round trip
    data = manager1.to_dict()
    manager2 = SplitManager.from_dict(data)
    
    # Compare structure
    assert manager1.root.orientation == manager2.root.orientation
    assert len(manager1.root.children) == len(manager2.root.children)
    
    # Check nested structure
    assert manager2.root.children[0].orientation == SplitOrientation.VERTICAL

def test_complex_nested_layout():
    """Test serialization of complex 3+ level layout."""
    manager = SplitManager()
    
    # Level 1
    id1 = manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    # Level 2
    child1 = manager.root.children[0]
    id2 = manager.split_node(child1.id, SplitOrientation.VERTICAL)
    
    # Level 3
    id3 = manager.split_node(id2, SplitOrientation.HORIZONTAL)
    
    # Serialize and deserialize
    data = manager.to_dict()
    manager2 = SplitManager.from_dict(data)
    
    # Should have 4 containers
    containers = manager2.get_all_containers()
    assert len(containers) == 4

def test_save_to_file():
    """Test saving layout to file."""
    manager = SplitManager()
    manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    try:
        LayoutState.save_to_file(manager, filepath)
        
        # Verify file exists and is valid JSON
        assert Path(filepath).exists()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert "root" in data
        assert "active_node_id" in data
    finally:
        Path(filepath).unlink()

def test_load_from_file():
    """Test loading layout from file."""
    manager1 = SplitManager()
    manager1.split_node(manager1.root.id, SplitOrientation.VERTICAL)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    try:
        LayoutState.save_to_file(manager1, filepath)
        manager2 = LayoutState.load_from_file(filepath)
        
        assert manager2 is not None
        assert manager2.root.orientation == SplitOrientation.VERTICAL
    finally:
        Path(filepath).unlink()

def test_load_missing_file():
    """Test loading from non-existent file."""
    manager = LayoutState.load_from_file("/non/existent/path.json")
    assert manager is None
