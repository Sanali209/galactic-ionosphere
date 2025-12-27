# -*- coding: utf-8 -*-
"""
Tests for NodeGraph Phase 2 - Flow Control and Variables

Tests cover:
- Event nodes (Start, Update, CustomEvent)
- Flow control nodes (Branch, Sequence, Loops)
- Variable nodes (Get, Set)
"""
import pytest

from src.nodegraph.nodes.events import (
    StartNode, UpdateNode, CustomEventNode, CallEventNode
)
from src.nodegraph.nodes.flow_control import (
    IfNode, BranchNode, SequenceNode, ForLoopNode,
    ForEachLoopNode, WhileLoopNode, DoOnceNode, GateNode, FlipFlopNode
)
from src.nodegraph.nodes.variables import (
    GetVariableNode, SetVariableNode, IncrementVariableNode
)
from src.nodegraph.nodes.utilities import PrintNode, MakeArrayNode
from src.nodegraph.core.pins import PinType, PinDirection
from src.nodegraph.core.graph import NodeGraph


class TestEventNodes:
    """Tests for event nodes."""
    
    def test_start_node_has_exec_output(self):
        """Start node should have exec output pin."""
        node = StartNode()
        assert "exec" in node.output_pins
        assert node.output_pins["exec"].pin_type == PinType.EXECUTION
    
    def test_update_node_has_delta_time(self):
        """Update node should output delta_time."""
        node = UpdateNode()
        assert "delta_time" in node.output_pins
        assert node.output_pins["delta_time"].pin_type == PinType.FLOAT
    
    def test_custom_event_stores_name(self):
        """CustomEvent should store event name."""
        node = CustomEventNode(event_name="OnPlayerDeath")
        assert node.event_name == "OnPlayerDeath"
    
    def test_custom_event_serialization(self):
        """CustomEvent should serialize event name."""
        node = CustomEventNode(event_name="TestEvent")
        data = node.to_dict()
        
        assert data["event_name"] == "TestEvent"
        
        restored = CustomEventNode.from_dict(data)
        assert restored.event_name == "TestEvent"


class TestFlowControlNodes:
    """Tests for flow control nodes."""
    
    def test_if_node_has_condition_and_then(self):
        """If node should have condition input and then output."""
        node = IfNode()
        assert "condition" in node.input_pins
        assert node.input_pins["condition"].pin_type == PinType.BOOLEAN
        assert "then" in node.output_pins
    
    def test_branch_node_has_true_and_false(self):
        """Branch node should have true and false outputs."""
        node = BranchNode()
        assert "true" in node.output_pins
        assert "false" in node.output_pins
    
    def test_sequence_node_has_multiple_outputs(self):
        """Sequence node should have multiple then outputs."""
        node = SequenceNode()
        assert "then_0" in node.output_pins
        assert "then_1" in node.output_pins
        assert "then_2" in node.output_pins
    
    def test_for_loop_has_index_output(self):
        """ForLoop should have index output."""
        node = ForLoopNode()
        assert "first_index" in node.input_pins
        assert "last_index" in node.input_pins
        assert "index" in node.output_pins
        assert "loop_body" in node.output_pins
        assert "completed" in node.output_pins
    
    def test_for_each_loop_has_element_output(self):
        """ForEachLoop should have element and index outputs."""
        node = ForEachLoopNode()
        assert "array" in node.input_pins
        assert "element" in node.output_pins
        assert "index" in node.output_pins
    
    def test_while_loop_has_condition(self):
        """WhileLoop should have condition input."""
        node = WhileLoopNode()
        assert "condition" in node.input_pins
        assert node.input_pins["condition"].default_value is True
    
    def test_gate_node_has_control_inputs(self):
        """Gate should have open/close/toggle inputs."""
        node = GateNode()
        assert "open" in node.input_pins
        assert "close" in node.input_pins
        assert "toggle" in node.input_pins
    
    def test_flipflop_has_a_and_b_outputs(self):
        """FlipFlop should have A and B outputs."""
        node = FlipFlopNode()
        assert "A" in node.output_pins
        assert "B" in node.output_pins
        assert "is_A" in node.output_pins


class TestVariableNodes:
    """Tests for variable nodes."""
    
    def test_get_variable_stores_name(self):
        """GetVariable should store variable name."""
        node = GetVariableNode(variable_name="health")
        assert node.variable_name == "health"
    
    def test_set_variable_has_value_input(self):
        """SetVariable should have value input."""
        node = SetVariableNode(variable_name="score")
        assert "value" in node.input_pins
        assert "exec" in node.input_pins
        assert "exec_out" in node.output_pins
    
    def test_variable_serialization(self):
        """Variable nodes should serialize variable name."""
        node = SetVariableNode(variable_name="playerName")
        data = node.to_dict()
        
        assert data["variable_name"] == "playerName"
        
        restored = SetVariableNode.from_dict(data)
        assert restored.variable_name == "playerName"
    
    def test_increment_variable_has_amount(self):
        """IncrementVariable should have amount input."""
        node = IncrementVariableNode(variable_name="counter")
        assert "amount" in node.input_pins
        assert node.input_pins["amount"].default_value == 1


class TestUtilityNodes:
    """Tests for utility nodes."""
    
    def test_print_node_has_message(self):
        """Print node should have message input."""
        node = PrintNode()
        assert "message" in node.input_pins
        assert node.input_pins["message"].pin_type == PinType.STRING
    
    def test_make_array_has_multiple_inputs(self):
        """MakeArray should have multiple item inputs."""
        node = MakeArrayNode()
        assert "item_0" in node.input_pins
        assert "item_1" in node.input_pins
        assert "array" in node.output_pins


class TestNodeConnections:
    """Tests for connecting Phase 2 nodes."""
    
    def test_connect_start_to_branch(self):
        """Should connect Start to Branch."""
        graph = NodeGraph()
        start = graph.add_node(StartNode())
        branch = graph.add_node(BranchNode())
        
        conn = graph.connect(start.node_id, "exec", branch.node_id, "exec")
        assert conn is not None
    
    def test_connect_branch_true_to_print(self):
        """Should connect Branch true output to Print."""
        graph = NodeGraph()
        branch = graph.add_node(BranchNode())
        print_node = graph.add_node(PrintNode())
        
        conn = graph.connect(branch.node_id, "true", print_node.node_id, "exec")
        assert conn is not None
    
    def test_connect_for_loop_to_sequence(self):
        """Should connect ForLoop body to Sequence."""
        graph = NodeGraph()
        loop = graph.add_node(ForLoopNode())
        seq = graph.add_node(SequenceNode())
        
        conn = graph.connect(loop.node_id, "loop_body", seq.node_id, "exec")
        assert conn is not None
