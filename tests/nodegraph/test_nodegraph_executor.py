# -*- coding: utf-8 -*-
"""
Tests for NodeGraph Phase 4 - Execution Engine

Tests cover:
- GraphExecutor basic flow
- Flow control execution (Branch, Sequence, Loops)
- Variable get/set
- Error handling
"""
import pytest
import asyncio

from src.nodegraph.core.graph import NodeGraph
from src.nodegraph.nodes.events import StartNode
from src.nodegraph.nodes.flow_control import (
    BranchNode, SequenceNode, ForLoopNode, IfNode
)
from src.nodegraph.nodes.variables import SetVariableNode, GetVariableNode
from src.nodegraph.nodes.utilities import PrintNode
from src.nodegraph.execution import GraphExecutor, ExecutionState


class TestGraphExecutorBasic:
    """Tests for basic execution."""
    
    def test_executor_creation(self):
        """Should create executor for graph."""
        graph = NodeGraph("Test")
        executor = GraphExecutor(graph)
        assert executor.graph == graph
        assert executor.state == ExecutionState.IDLE
    
    def test_simple_start_execution(self):
        """Should execute from Start node."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        print_node = graph.add_node(PrintNode())
        print_node._input_pins["message"].default_value = "Hello!"
        
        graph.connect(start.node_id, "exec", print_node.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
        assert len(context.logs) > 0
    
    def test_no_start_node_error(self):
        """Should error if no start node."""
        graph = NodeGraph("Test")
        # No start node added
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.ERROR
        assert "entry point" in context.error.lower()


class TestFlowControlExecution:
    """Tests for flow control execution."""
    
    def test_branch_true(self):
        """Should follow True branch when condition is True."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        branch = graph.add_node(BranchNode())
        branch._input_pins["condition"].default_value = True
        
        print_true = graph.add_node(PrintNode())
        print_true._input_pins["message"].default_value = "TRUE"
        
        graph.connect(start.node_id, "exec", branch.node_id, "exec")
        graph.connect(branch.node_id, "true", print_true.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
        # Check logs contain "TRUE"
        log_messages = [log.message for log in context.logs]
        assert any("TRUE" in msg for msg in log_messages)
    
    def test_branch_false(self):
        """Should follow False branch when condition is False."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        branch = graph.add_node(BranchNode())
        branch._input_pins["condition"].default_value = False
        
        print_false = graph.add_node(PrintNode())
        print_false._input_pins["message"].default_value = "FALSE"
        
        graph.connect(start.node_id, "exec", branch.node_id, "exec")
        graph.connect(branch.node_id, "false", print_false.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
        log_messages = [log.message for log in context.logs]
        assert any("FALSE" in msg for msg in log_messages)
    
    def test_sequence_order(self):
        """Should execute sequence outputs in order."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        seq = graph.add_node(SequenceNode())
        
        graph.connect(start.node_id, "exec", seq.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
    
    def test_for_loop_iterations(self):
        """Should execute loop body for each iteration."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        loop = graph.add_node(ForLoopNode())
        loop._input_pins["first_index"].default_value = 1
        loop._input_pins["last_index"].default_value = 3
        
        print_node = graph.add_node(PrintNode())
        print_node._input_pins["message"].default_value = "Loop"
        
        graph.connect(start.node_id, "exec", loop.node_id, "exec")
        graph.connect(loop.node_id, "loop_body", print_node.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
        # Should have multiple "Loop" prints
        loop_logs = [log for log in context.logs if "Loop" in log.message]
        assert len(loop_logs) >= 3


class TestVariableExecution:
    """Tests for variable operations."""
    
    def test_set_variable(self):
        """Should set variable value."""
        graph = NodeGraph("Test")
        graph.add_variable("myvar", "Integer", 0)
        
        start = graph.add_node(StartNode())
        set_var = graph.add_node(SetVariableNode(variable_name="myvar"))
        set_var._input_pins["value"].default_value = 42
        
        graph.connect(start.node_id, "exec", set_var.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert executor.state == ExecutionState.COMPLETED
        assert context.get_variable("myvar") == 42
    
    def test_variable_persistence(self):
        """Variables should persist across nodes."""
        graph = NodeGraph("Test")
        graph.add_variable("counter", "Integer", 0)
        
        start = graph.add_node(StartNode())
        set_var = graph.add_node(SetVariableNode(variable_name="counter"))
        set_var._input_pins["value"].default_value = 10
        
        print_node = graph.add_node(PrintNode())
        print_node._input_pins["message"].default_value = "Done"
        
        graph.connect(start.node_id, "exec", set_var.node_id, "exec")
        graph.connect(set_var.node_id, "exec_out", print_node.node_id, "exec")
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert context.get_variable("counter") == 10


class TestExecutionContext:
    """Tests for ExecutionContext."""
    
    def test_context_logs(self):
        """Context should collect logs."""
        graph = NodeGraph("Test")
        start = graph.add_node(StartNode())
        
        executor = GraphExecutor(graph)
        context = executor.run()
        
        assert len(context.logs) > 0
        assert all(log.timestamp > 0 for log in context.logs)
    
    def test_context_error_tracking(self):
        """Context should track errors."""
        from src.nodegraph.execution.executor import ExecutionContext
        from src.nodegraph.nodes.utilities import PrintNode
        
        context = ExecutionContext()
        node = PrintNode()
        
        context.set_error(node, "Test error", "message")
        
        assert context.error == "Test error"
        assert context.error_node_id == node.node_id
        assert context.error_pin_name == "message"
        assert node.has_error
