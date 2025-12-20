# -*- coding: utf-8 -*-
"""
Phase 4 Demo - Graph Execution Engine

This demo shows:
1. Creating a graph programmatically
2. Running the graph with GraphExecutor
3. Observing execution flow
4. Working with variables
5. Handling loops and branches

Run with: py demos/phase4_execution/demo.py
"""
import sys
from pathlib import Path

# Add foundation to path
FOUNDATION_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FOUNDATION_DIR))

from src.nodegraph.core.graph import NodeGraph
from src.nodegraph.nodes.events import StartNode
from src.nodegraph.nodes.flow_control import (
    BranchNode, SequenceNode, ForLoopNode
)
from src.nodegraph.nodes.variables import SetVariableNode
from src.nodegraph.nodes.utilities import PrintNode
from src.nodegraph.execution import GraphExecutor, ExecutionState


def demo_simple_flow():
    """Demonstrate simple linear execution."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Linear Flow")
    print("=" * 60)
    
    graph = NodeGraph("Simple Flow")
    
    # Create nodes
    start = graph.add_node(StartNode())
    print1 = graph.add_node(PrintNode())
    print1._input_pins["message"].default_value = "Hello, NodeGraph!"
    
    print2 = graph.add_node(PrintNode())
    print2._input_pins["message"].default_value = "Execution complete!"
    
    # Connect
    graph.connect(start.node_id, "exec", print1.node_id, "exec")
    graph.connect(print1.node_id, "exec_out", print2.node_id, "exec")
    
    # Execute
    print("\nExecuting graph...")
    executor = GraphExecutor(graph)
    context = executor.run()
    
    print(f"\nState: {executor.state.name}")
    print(f"Logs: {len(context.logs)} entries")
    
    return executor.state == ExecutionState.COMPLETED


def demo_branching():
    """Demonstrate conditional branching."""
    print("\n" + "=" * 60)
    print("Demo 2: Conditional Branching")
    print("=" * 60)
    
    graph = NodeGraph("Branching")
    
    # Create nodes
    start = graph.add_node(StartNode())
    branch = graph.add_node(BranchNode())
    branch._input_pins["condition"].default_value = True  # Will take True path
    
    print_true = graph.add_node(PrintNode())
    print_true._input_pins["message"].default_value = "✓ Took the TRUE path!"
    
    print_false = graph.add_node(PrintNode())
    print_false._input_pins["message"].default_value = "✗ Took the FALSE path!"
    
    # Connect
    graph.connect(start.node_id, "exec", branch.node_id, "exec")
    graph.connect(branch.node_id, "true", print_true.node_id, "exec")
    graph.connect(branch.node_id, "false", print_false.node_id, "exec")
    
    # Execute
    print("\nExecuting with condition=True...")
    executor = GraphExecutor(graph)
    context = executor.run()
    
    print(f"\nState: {executor.state.name}")
    
    # Now try with False
    branch._input_pins["condition"].default_value = False
    print("\nExecuting with condition=False...")
    executor2 = GraphExecutor(graph)
    context2 = executor2.run()
    
    return True


def demo_loop():
    """Demonstrate loop execution."""
    print("\n" + "=" * 60)
    print("Demo 3: For Loop")
    print("=" * 60)
    
    graph = NodeGraph("Loop Demo")
    
    # Create nodes
    start = graph.add_node(StartNode())
    loop = graph.add_node(ForLoopNode())
    loop._input_pins["first_index"].default_value = 1
    loop._input_pins["last_index"].default_value = 5
    
    print_loop = graph.add_node(PrintNode())
    print_loop._input_pins["message"].default_value = "  Loop iteration..."
    
    print_done = graph.add_node(PrintNode())
    print_done._input_pins["message"].default_value = "Loop completed!"
    
    # Connect
    graph.connect(start.node_id, "exec", loop.node_id, "exec")
    graph.connect(loop.node_id, "loop_body", print_loop.node_id, "exec")
    graph.connect(loop.node_id, "completed", print_done.node_id, "exec")
    
    # Execute
    print("\nExecuting loop (1 to 5)...")
    executor = GraphExecutor(graph)
    context = executor.run()
    
    print(f"\nState: {executor.state.name}")
    print(f"Total log entries: {len(context.logs)}")
    
    return executor.state == ExecutionState.COMPLETED


def demo_variables():
    """Demonstrate variable operations."""
    print("\n" + "=" * 60)
    print("Demo 4: Variables")
    print("=" * 60)
    
    graph = NodeGraph("Variables Demo")
    
    # Add graph variable
    graph.add_variable("score", "Integer", 0)
    
    # Create nodes
    start = graph.add_node(StartNode())
    
    set_var = graph.add_node(SetVariableNode(variable_name="score"))
    set_var._input_pins["value"].default_value = 100
    
    print_node = graph.add_node(PrintNode())
    print_node._input_pins["message"].default_value = "Variable set!"
    
    # Connect
    graph.connect(start.node_id, "exec", set_var.node_id, "exec")
    graph.connect(set_var.node_id, "exec_out", print_node.node_id, "exec")
    
    # Execute
    print("\nExecuting variable operations...")
    executor = GraphExecutor(graph)
    
    # Track variable changes
    changes = []
    executor.on_variable_changed(lambda name, val: changes.append((name, val)))
    
    context = executor.run()
    
    print(f"\nState: {executor.state.name}")
    print(f"Variable 'score': {context.get_variable('score')}")
    print(f"Variable changes recorded: {changes}")
    
    return context.get_variable("score") == 100


def demo_sequence():
    """Demonstrate sequence execution."""
    print("\n" + "=" * 60)
    print("Demo 5: Sequence (Ordered Execution)")
    print("=" * 60)
    
    graph = NodeGraph("Sequence Demo")
    
    # Create nodes
    start = graph.add_node(StartNode())
    seq = graph.add_node(SequenceNode())
    
    prints = []
    for i in range(3):
        p = graph.add_node(PrintNode())
        p._input_pins["message"].default_value = f"Step {i + 1}"
        prints.append(p)
    
    # Connect
    graph.connect(start.node_id, "exec", seq.node_id, "exec")
    for i, p in enumerate(prints):
        graph.connect(seq.node_id, f"then_{i}", p.node_id, "exec")
    
    # Execute
    print("\nExecuting sequence...")
    executor = GraphExecutor(graph)
    context = executor.run()
    
    print(f"\nState: {executor.state.name}")
    
    return executor.state == ExecutionState.COMPLETED


def main():
    print("=" * 60)
    print("NodeGraph Phase 4 Demo - Execution Engine")
    print("=" * 60)
    
    results = []
    
    results.append(("Simple Flow", demo_simple_flow()))
    results.append(("Branching", demo_branching()))
    results.append(("Loop", demo_loop()))
    results.append(("Variables", demo_variables()))
    results.append(("Sequence", demo_sequence()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All demos completed successfully!")
    else:
        print("Some demos failed. Check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
