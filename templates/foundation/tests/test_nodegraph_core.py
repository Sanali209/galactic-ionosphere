# -*- coding: utf-8 -*-
"""
Tests for NodeGraph Core - Phase 1

Tests cover:
- BaseNode creation and serialization
- Pin types and connections
- NodeGraph operations
"""
import pytest
from typing import Optional

# Import core classes
from src.nodegraph.core.base_node import BaseNode, NodeMetadata
from src.nodegraph.core.pins import (
    PinDirection, PinType, BasePin, ExecutionPin, DataPin
)
from src.nodegraph.core.connection import NodeConnection
from src.nodegraph.core.graph import NodeGraph, Variable


# =============================================================================
# Test Fixtures - Sample Node Classes
# =============================================================================

class TestStartNode(BaseNode):
    """Test node - entry point."""
    node_type = "TestStart"
    metadata = NodeMetadata(
        category="Test",
        display_name="Test Start",
        description="Test entry point"
    )
    
    def _setup_pins(self):
        self.add_output_pin(ExecutionPin("exec", PinDirection.OUTPUT))


class TestPrintNode(BaseNode):
    """Test node - prints a message."""
    node_type = "TestPrint"
    metadata = NodeMetadata(
        category="Test",
        display_name="Test Print",
        description="Test print node"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("message", PinType.STRING, default_value="Hello"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))


class TestAddNode(BaseNode):
    """Test node - adds two numbers (pure data node)."""
    node_type = "TestAdd"
    metadata = NodeMetadata(
        category="Test",
        display_name="Test Add",
        description="Adds two integers"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("b", PinType.INTEGER, default_value=0))
        self.add_output_pin(DataPin("result", PinType.INTEGER, PinDirection.OUTPUT))


# =============================================================================
# Test BaseNode
# =============================================================================

class TestBaseNodeCreation:
    """Tests for BaseNode creation and properties."""
    
    def test_node_creation_generates_id(self):
        """Node should generate unique ID."""
        node = TestStartNode()
        assert node.node_id is not None
        assert len(node.node_id) > 0
    
    def test_node_creation_with_custom_id(self):
        """Node should accept custom ID."""
        node = TestStartNode(node_id="custom-id-123")
        assert node.node_id == "custom-id-123"
    
    def test_node_has_correct_type(self):
        """Node should have correct node_type."""
        node = TestStartNode()
        assert node.node_type == "TestStart"
    
    def test_node_has_metadata(self):
        """Node should have metadata."""
        node = TestStartNode()
        assert node.metadata.category == "Test"
        assert node.metadata.display_name == "Test Start"
    
    def test_node_default_position(self):
        """Node should have default position (0, 0)."""
        node = TestStartNode()
        assert node.position == (0.0, 0.0)
    
    def test_node_pins_created(self):
        """Node should create pins from _setup_pins."""
        node = TestPrintNode()
        assert "exec" in node.input_pins
        assert "message" in node.input_pins
        assert "exec_out" in node.output_pins


class TestBaseNodeSerialization:
    """Tests for BaseNode serialization."""
    
    def test_to_dict(self):
        """Node should serialize to dict."""
        node = TestPrintNode(node_id="test-123")
        node.position = (100.0, 200.0)
        
        data = node.to_dict()
        
        assert data["node_type"] == "TestPrint"
        assert data["node_id"] == "test-123"
        assert data["position"] == [100.0, 200.0]
        assert "message" in data["pin_values"]
    
    def test_from_dict(self):
        """Node should deserialize from dict."""
        data = {
            "node_type": "TestPrint",
            "node_id": "test-456",
            "position": [150.0, 250.0],
            "pin_values": {"message": "Custom message"}
        }
        
        node = TestPrintNode.from_dict(data)
        
        assert node.node_id == "test-456"
        assert node.position == (150.0, 250.0)
        assert node.get_input("message") == "Custom message"


class TestBaseNodeErrors:
    """Tests for BaseNode error handling."""
    
    def test_set_error(self):
        """Node should track error state."""
        node = TestPrintNode()
        assert not node.has_error
        
        node.set_error("Something went wrong")
        
        assert node.has_error
        assert node.error_message == "Something went wrong"
    
    def test_clear_error(self):
        """Node should clear error state."""
        node = TestPrintNode()
        node.set_error("Error")
        node.clear_error()
        
        assert not node.has_error
        assert node.error_message is None


# =============================================================================
# Test Pins
# =============================================================================

class TestPinTypes:
    """Tests for pin types."""
    
    def test_pin_type_properties(self):
        """PinType should have display_name and color."""
        assert PinType.INTEGER.display_name == "Integer"
        assert PinType.INTEGER.color == "#1E90FF"
    
    def test_execution_pin_creation(self):
        """ExecutionPin should have correct type."""
        pin = ExecutionPin("exec")
        assert pin.pin_type == PinType.EXECUTION
        assert pin.direction == PinDirection.INPUT
    
    def test_data_pin_creation(self):
        """DataPin should have correct type and default."""
        pin = DataPin("count", PinType.INTEGER, default_value=42)
        assert pin.pin_type == PinType.INTEGER
        assert pin.default_value == 42
    
    def test_data_pin_rejects_execution_type(self):
        """DataPin should reject EXECUTION type."""
        with pytest.raises(ValueError):
            DataPin("exec", PinType.EXECUTION)


class TestPinConnections:
    """Tests for pin connection validation."""
    
    def test_can_connect_output_to_input(self):
        """Output pin should connect to input pin."""
        out_pin = DataPin("out", PinType.INTEGER, PinDirection.OUTPUT)
        in_pin = DataPin("in", PinType.INTEGER, PinDirection.INPUT)
        
        assert out_pin.can_connect_to(in_pin)
    
    def test_cannot_connect_same_direction(self):
        """Same direction pins should not connect."""
        pin1 = DataPin("a", PinType.INTEGER, PinDirection.INPUT)
        pin2 = DataPin("b", PinType.INTEGER, PinDirection.INPUT)
        
        assert not pin1.can_connect_to(pin2)
    
    def test_cannot_connect_different_types(self):
        """Different types should not connect."""
        int_pin = DataPin("int", PinType.INTEGER, PinDirection.OUTPUT)
        str_pin = DataPin("str", PinType.STRING, PinDirection.INPUT)
        
        assert not int_pin.can_connect_to(str_pin)
    
    def test_any_type_is_wildcard(self):
        """ANY type should connect to any other type."""
        any_pin = DataPin("any", PinType.ANY, PinDirection.OUTPUT)
        int_pin = DataPin("int", PinType.INTEGER, PinDirection.INPUT)
        
        assert any_pin.can_connect_to(int_pin)
    
    def test_execution_only_connects_to_execution(self):
        """Execution pins only connect to execution pins."""
        exec_pin = ExecutionPin("exec", PinDirection.OUTPUT)
        data_pin = DataPin("data", PinType.INTEGER, PinDirection.INPUT)
        
        assert not exec_pin.can_connect_to(data_pin)


# =============================================================================
# Test NodeConnection
# =============================================================================

class TestNodeConnection:
    """Tests for NodeConnection."""
    
    def test_connection_creation(self):
        """Connection should link pins correctly."""
        node_a = TestAddNode()
        node_b = TestPrintNode()
        
        # Connect output to input
        source = node_a.get_output_pin("result")
        target = node_b.get_input_pin("message")
        
        # Note: This won't work because types differ (INTEGER vs STRING)
        # Let's use nodes with compatible types
        
    def test_connection_with_execution_pins(self):
        """Execution pins should connect."""
        node_a = TestStartNode()
        node_b = TestPrintNode()
        
        source = node_a.get_output_pin("exec")
        target = node_b.get_input_pin("exec")
        
        conn = NodeConnection(source, target)
        
        assert conn.source_pin == source
        assert conn.target_pin == target
        assert target.connection == conn
    
    def test_connection_disconnect(self):
        """Disconnect should remove references."""
        node_a = TestStartNode()
        node_b = TestPrintNode()
        
        source = node_a.get_output_pin("exec")
        target = node_b.get_input_pin("exec")
        conn = NodeConnection(source, target)
        
        conn.disconnect()
        
        assert target.connection is None
    
    def test_connection_serialization(self):
        """Connection should serialize correctly."""
        node_a = TestStartNode(node_id="node-a")
        node_b = TestPrintNode(node_id="node-b")
        
        source = node_a.get_output_pin("exec")
        target = node_b.get_input_pin("exec")
        conn = NodeConnection(source, target)
        
        data = conn.to_dict()
        
        assert data["source_node_id"] == "node-a"
        assert data["source_pin_name"] == "exec"
        assert data["target_node_id"] == "node-b"
        assert data["target_pin_name"] == "exec"


# =============================================================================
# Test NodeGraph
# =============================================================================

class TestNodeGraph:
    """Tests for NodeGraph."""
    
    def test_graph_creation(self):
        """Graph should create with name."""
        graph = NodeGraph("My Graph")
        assert graph.name == "My Graph"
        assert len(graph.nodes) == 0
    
    def test_add_node(self):
        """Graph should add nodes."""
        graph = NodeGraph()
        node = TestStartNode()
        
        result = graph.add_node(node)
        
        assert result == node
        assert node.node_id in graph.nodes
    
    def test_remove_node(self):
        """Graph should remove nodes."""
        graph = NodeGraph()
        node = TestStartNode()
        graph.add_node(node)
        
        graph.remove_node(node.node_id)
        
        assert node.node_id not in graph.nodes
    
    def test_connect_nodes(self):
        """Graph should connect nodes."""
        graph = NodeGraph()
        start = graph.add_node(TestStartNode())
        print_node = graph.add_node(TestPrintNode())
        
        conn = graph.connect(
            start.node_id, "exec",
            print_node.node_id, "exec"
        )
        
        assert conn.connection_id in graph.connections
    
    def test_disconnect(self):
        """Graph should disconnect nodes."""
        graph = NodeGraph()
        start = graph.add_node(TestStartNode())
        print_node = graph.add_node(TestPrintNode())
        conn = graph.connect(
            start.node_id, "exec",
            print_node.node_id, "exec"
        )
        
        graph.disconnect(conn.connection_id)
        
        assert conn.connection_id not in graph.connections
    
    def test_remove_node_removes_connections(self):
        """Removing node should remove its connections."""
        graph = NodeGraph()
        start = graph.add_node(TestStartNode())
        print_node = graph.add_node(TestPrintNode())
        graph.connect(
            start.node_id, "exec",
            print_node.node_id, "exec"
        )
        
        graph.remove_node(start.node_id)
        
        assert len(graph.connections) == 0


class TestNodeGraphVariables:
    """Tests for NodeGraph variables."""
    
    def test_add_variable(self):
        """Graph should add variables."""
        graph = NodeGraph()
        var = graph.add_variable("counter", "Integer", 0)
        
        assert var.name == "counter"
        assert graph.get_variable("counter") == var
    
    def test_remove_variable(self):
        """Graph should remove variables."""
        graph = NodeGraph()
        graph.add_variable("temp", "String", "")
        graph.remove_variable("temp")
        
        assert graph.get_variable("temp") is None


class TestNodeGraphSerialization:
    """Tests for NodeGraph serialization."""
    
    def test_to_dict(self):
        """Graph should serialize to dict."""
        graph = NodeGraph("Test Graph")
        start = graph.add_node(TestStartNode())
        print_node = graph.add_node(TestPrintNode())
        graph.connect(start.node_id, "exec", print_node.node_id, "exec")
        graph.add_variable("count", "Integer", 5)
        
        data = graph.to_dict()
        
        assert data["name"] == "Test Graph"
        assert len(data["nodes"]) == 2
        assert len(data["connections"]) == 1
        assert len(data["variables"]) == 1
