"""
Unit tests for SplitManager (Phase 1).
Testing split tree creation, manipulation, and serialization.
"""
import pytest
from src.ui.documents.split_manager import SplitManager, SplitOrientation, SplitNode

def test_split_tree_creation_horizontal():
    """Test creating horizontal split."""
    manager = SplitManager()
    
    # Start with single container
    assert manager.root.is_container
    assert len(manager.root.children) == 0
    
    # Split horizontally
    new_id = manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    assert new_id is not None
    assert not manager.root.is_container  # Root is now a splitter
    assert len(manager.root.children) == 2
    assert manager.root.orientation == SplitOrientation.HORIZONTAL

def test_split_tree_creation_vertical():
    """Test creating vertical split."""
    manager = SplitManager()
    new_id = manager.split_node(manager.root.id, SplitOrientation.VERTICAL)
    
    assert new_id is not None
    assert manager.root.orientation == SplitOrientation.VERTICAL

def test_split_removal_and_merge():
    """Test removing split and merging with sibling."""
    manager = SplitManager()
    
    # Create split
    second_id = manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    child1_id = manager.root.children[0].id
    child2_id = manager.root.children[1].id
    
    assert len(manager.root.children) == 2
    
    # Remove second child
    success = manager.remove_split(child2_id)
    
    assert success
    # Root should be replaced by remaining child
    assert manager.root.id == child1_id
    assert manager.root.is_container

def test_split_navigation():
    """Test finding nodes by ID."""
    manager = SplitManager()
    
    root_id = manager.root.id
    second_id = manager.split_node(root_id, SplitOrientation.HORIZONTAL)
    
    # Should find both children
    child1 = manager.get_node(manager.root.children[0].id)
    child2 = manager.get_node(second_id)
    
    assert child1 is not None
    assert child2 is not None
    assert child1.is_container
    assert child2.is_container

def test_split_resizing():
    """Test split sizes (ratios)."""
    manager = SplitManager()
    manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    # Default should be equal split
    assert manager.root.sizes == [1, 1]
    
    # Modify sizes
    manager.root.sizes = [2, 1]  # 2:1 ratio
    assert manager.root.sizes == [2, 1]

def test_edge_case_single_split():
    """Test behavior with single container (no splits)."""
    manager = SplitManager()
    
    containers = manager.get_all_containers()
    assert len(containers) == 1
    assert containers[0] == manager.root

def test_edge_case_empty_splits():
    """Test cannot split non-container node."""
    manager = SplitManager()
    manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    # Try to split the splitter (should fail)
    result = manager.split_node(manager.root.id, SplitOrientation.VERTICAL)
    assert result is None

def test_nested_splits():
    """Test creating nested splits (3 levels)."""
    manager = SplitManager()
    
    # Level 1: Split root horizontally
    second_id = manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    
    # Level 2: Split first child vertically
    first_child = manager.root.children[0]
    third_id = manager.split_node(first_child.id, SplitOrientation.VERTICAL)
    
    # Level 3: Split one of those vertically again
    fourth_id = manager.split_node(third_id, SplitOrientation.VERTICAL)
    
    assert fourth_id is not None
    
    # Should have 4 containers total
    containers = manager.get_all_containers()
    assert len(containers) == 4

def test_get_all_containers():
    """Test retrieving all leaf container nodes."""
    manager = SplitManager()
    
    # Create 3 splits
    second_id = manager.split_node(manager.root.id, SplitOrientation.HORIZONTAL)
    third_id = manager.split_node(second_id, SplitOrientation.VERTICAL)
    
    containers = manager.get_all_containers()
    assert len(containers) == 3
    
    # All should be containers
    for container in containers:
        assert container.is_container
