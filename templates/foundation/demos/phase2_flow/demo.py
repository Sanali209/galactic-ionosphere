# -*- coding: utf-8 -*-
"""
Phase 2 Demo - Flow Control and Variables

This demo shows:
1. Creating graphs with flow control nodes
2. If/Branch conditional logic
3. Sequence node for ordered execution
4. Loop nodes (ForLoop, ForEach)
5. Variable Get/Set nodes

Run with: py demos/phase2_flow/demo.py
"""
import sys
from pathlib import Path

# Add foundation to path
FOUNDATION_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FOUNDATION_DIR))

from src.nodegraph.core.graph import NodeGraph
from src.nodegraph.nodes.events import StartNode
from src.nodegraph.nodes.flow_control import (
    IfNode, BranchNode, SequenceNode, ForLoopNode, 
    ForEachLoopNode, FlipFlopNode
)
from src.nodegraph.nodes.variables import (
    GetVariableNode, SetVariableNode, IncrementVariableNode
)
from src.nodegraph.nodes.utilities import PrintNode


def demo_conditional_flow():
    """Demonstrate If and Branch nodes."""
    print("\n" + "=" * 60)
    print("Demo 1: Conditional Flow (If/Branch)")
    print("=" * 60)
    
    graph = NodeGraph("Conditional Demo")
    
    # Create nodes
    start = graph.add_node(StartNode())
    branch = graph.add_node(BranchNode())
    if_node = graph.add_node(IfNode())
    print_true = graph.add_node(PrintNode())
    print_false = graph.add_node(PrintNode())
    
    # Set default values
    branch._input_pins["condition"].default_value = True
    print_true._input_pins["message"].default_value = "Branch took TRUE path!"
    print_false._input_pins["message"].default_value = "Branch took FALSE path!"
    
    # Connect nodes
    graph.connect(start.node_id, "exec", branch.node_id, "exec")
    graph.connect(branch.node_id, "true", print_true.node_id, "exec")
    graph.connect(branch.node_id, "false", print_false.node_id, "exec")
    
    print(f"Created graph: {graph}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Connections: {len(graph.connections)}")
    print("\nGraph structure:")
    print("  Start -> Branch")
    print("             |-- true --> Print('Branch took TRUE path!')")
    print("             |-- false --> Print('Branch took FALSE path!')")
    
    return graph


def demo_sequence():
    """Demonstrate Sequence node."""
    print("\n" + "=" * 60)
    print("Demo 2: Sequence Execution")
    print("=" * 60)
    
    graph = NodeGraph("Sequence Demo")
    
    # Create nodes
    start = graph.add_node(StartNode())
    seq = graph.add_node(SequenceNode())
    print0 = graph.add_node(PrintNode())
    print1 = graph.add_node(PrintNode())
    print2 = graph.add_node(PrintNode())
    
    # Set messages
    print0._input_pins["message"].default_value = "Step 1: Initialize"
    print1._input_pins["message"].default_value = "Step 2: Process"
    print2._input_pins["message"].default_value = "Step 3: Complete"
    
    # Connect
    graph.connect(start.node_id, "exec", seq.node_id, "exec")
    graph.connect(seq.node_id, "then_0", print0.node_id, "exec")
    graph.connect(seq.node_id, "then_1", print1.node_id, "exec")
    graph.connect(seq.node_id, "then_2", print2.node_id, "exec")
    
    print(f"Created graph: {graph}")
    print("\nGraph structure:")
    print("  Start -> Sequence")
    print("             |-- then_0 --> Print('Step 1: Initialize')")
    print("             |-- then_1 --> Print('Step 2: Process')")
    print("             |-- then_2 --> Print('Step 3: Complete')")
    
    return graph


def demo_for_loop():
    """Demonstrate ForLoop node."""
    print("\n" + "=" * 60)
    print("Demo 3: For Loop")
    print("=" * 60)
    
    graph = NodeGraph("ForLoop Demo")
    
    # Create nodes
    start = graph.add_node(StartNode())
    loop = graph.add_node(ForLoopNode())
    print_in_loop = graph.add_node(PrintNode())
    print_done = graph.add_node(PrintNode())
    
    # Configure loop
    loop._input_pins["first_index"].default_value = 1
    loop._input_pins["last_index"].default_value = 5
    print_in_loop._input_pins["message"].default_value = "Loop iteration"
    print_done._input_pins["message"].default_value = "Loop completed!"
    
    # Connect
    graph.connect(start.node_id, "exec", loop.node_id, "exec")
    graph.connect(loop.node_id, "loop_body", print_in_loop.node_id, "exec")
    graph.connect(loop.node_id, "completed", print_done.node_id, "exec")
    
    print(f"Created graph: {graph}")
    print("\nGraph structure:")
    print("  Start -> ForLoop(first=1, last=5)")
    print("             |-- loop_body --> Print('Loop iteration')")
    print("             |               (index output: 1, 2, 3, 4, 5)")
    print("             |-- completed --> Print('Loop completed!')")
    
    return graph


def demo_variables():
    """Demonstrate variable nodes."""
    print("\n" + "=" * 60)
    print("Demo 4: Variables")
    print("=" * 60)
    
    graph = NodeGraph("Variables Demo")
    
    # Add graph variable
    counter = graph.add_variable("counter", "Integer", 0)
    print(f"Created variable: {counter.name} = {counter.default_value}")
    
    # Create nodes
    start = graph.add_node(StartNode())
    seq = graph.add_node(SequenceNode())
    set_var = graph.add_node(SetVariableNode(variable_name="counter"))
    inc_var = graph.add_node(IncrementVariableNode(variable_name="counter"))
    get_var = graph.add_node(GetVariableNode(variable_name="counter"))
    print_val = graph.add_node(PrintNode())
    
    # Set initial value
    set_var._input_pins["value"].default_value = 10
    
    # Connect
    graph.connect(start.node_id, "exec", seq.node_id, "exec")
    graph.connect(seq.node_id, "then_0", set_var.node_id, "exec")
    graph.connect(seq.node_id, "then_1", inc_var.node_id, "exec")
    graph.connect(seq.node_id, "then_2", print_val.node_id, "exec")
    # Connect data: get_var.value -> print.message (would need type conversion in real use)
    
    print(f"Created graph: {graph}")
    print("\nGraph structure:")
    print("  Start -> Sequence")
    print("             |-- then_0 --> SetVariable('counter', 10)")
    print("             |-- then_1 --> IncrementVariable('counter', +1)")
    print("             |-- then_2 --> Print (show counter value: 11)")
    print(f"\nVariables: {list(graph.variables.keys())}")
    
    return graph


def main():
    print("=" * 60)
    print("NodeGraph Phase 2 Demo - Flow Control & Variables")
    print("=" * 60)
    
    # Run all demos
    graph1 = demo_conditional_flow()
    graph2 = demo_sequence()
    graph3 = demo_for_loop()
    graph4 = demo_variables()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nCreated 4 demo graphs:")
    print(f"  1. Conditional: {len(graph1.nodes)} nodes, {len(graph1.connections)} connections")
    print(f"  2. Sequence:    {len(graph2.nodes)} nodes, {len(graph2.connections)} connections")
    print(f"  3. ForLoop:     {len(graph3.nodes)} nodes, {len(graph3.connections)} connections")
    print(f"  4. Variables:   {len(graph4.nodes)} nodes, {len(graph4.connections)} connections")
    
    # Count all node types
    from src.nodegraph.nodes import ALL_NODES
    print(f"\nTotal built-in node types available: {len(ALL_NODES)}")
    
    # Show categories
    categories = {}
    for node_cls in ALL_NODES:
        cat = node_cls.metadata.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node_cls.node_type)
    
    print("\nNode categories:")
    for cat, nodes in sorted(categories.items()):
        print(f"  {cat}: {len(nodes)} nodes")
        for name in nodes[:3]:
            print(f"    - {name}")
        if len(nodes) > 3:
            print(f"    ... and {len(nodes) - 3} more")
    
    print("\n" + "=" * 60)
    print("Phase 2 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
