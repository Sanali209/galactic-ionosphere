# -*- coding: utf-8 -*-
"""
Phase 1 Demo - Core Node System

This demo shows how to:
1. Create custom nodes by inheriting from BaseNode
2. Create a NodeGraph and add nodes
3. Connect nodes together
4. Serialize and deserialize graphs

Run with: py demos/phase1_core/demo.py
"""
import json
import sys
from pathlib import Path

# Add foundation to path
FOUNDATION_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FOUNDATION_DIR))

from src.nodegraph.core.base_node import BaseNode, NodeMetadata
from src.nodegraph.core.pins import ExecutionPin, DataPin, PinType, PinDirection
from src.nodegraph.core.graph import NodeGraph


# =============================================================================
# Custom Node Definitions
# =============================================================================

class StartNode(BaseNode):
    """Entry point node - execution begins here."""
    node_type = "Start"
    metadata = NodeMetadata(
        category="Events",
        display_name="Start",
        description="Entry point for graph execution",
        color="#CC0000"
    )
    
    def _setup_pins(self):
        self.add_output_pin(ExecutionPin("exec", PinDirection.OUTPUT))


class PrintNode(BaseNode):
    """Print a message to console."""
    node_type = "Print"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="Print",
        description="Print message to console",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("message", PinType.STRING, default_value="Hello World!"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))


class AddNode(BaseNode):
    """Add two integers (pure data node)."""
    node_type = "Add"
    metadata = NodeMetadata(
        category="Math",
        display_name="Add",
        description="Add two integers",
        color="#00CC66"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("b", PinType.INTEGER, default_value=0))
        self.add_output_pin(DataPin("result", PinType.INTEGER, PinDirection.OUTPUT))


class SetVariableNode(BaseNode):
    """Set a variable value."""
    node_type = "SetVariable"
    metadata = NodeMetadata(
        category="Variables",
        display_name="Set Variable",
        description="Set variable value",
        color="#FFD700"
    )
    
    def __init__(self, node_id=None, variable_name: str = "MyVar"):
        self.variable_name = variable_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("new_value", PinType.ANY, PinDirection.OUTPUT))


# =============================================================================
# Demo Script
# =============================================================================

def main():
    print("=" * 60)
    print("NodeGraph Phase 1 Demo - Core Node System")
    print("=" * 60)
    
    # 1. Create a new graph
    print("\n1. Creating NodeGraph...")
    graph = NodeGraph("Hello World Graph")
    print(f"   Created: {graph}")
    
    # 2. Create and add nodes
    print("\n2. Adding nodes...")
    start = StartNode()
    start.position = (100, 100)
    graph.add_node(start)
    print(f"   Added: {start} at position {start.position}")
    
    print1 = PrintNode()
    print1.position = (300, 100)
    print1._input_pins["message"].default_value = "Hello from NodeGraph!"
    graph.add_node(print1)
    print(f"   Added: {print1}")
    
    print2 = PrintNode()
    print2.position = (500, 100)
    print2._input_pins["message"].default_value = "Goodbye!"
    graph.add_node(print2)
    print(f"   Added: {print2}")
    
    add = AddNode()
    add.position = (300, 250)
    add._input_pins["a"].default_value = 10
    add._input_pins["b"].default_value = 25
    graph.add_node(add)
    print(f"   Added: {add}")
    
    # 3. Connect nodes
    print("\n3. Connecting nodes...")
    conn1 = graph.connect(start.node_id, "exec", print1.node_id, "exec")
    print(f"   Connected: Start -> Print1 ({conn1})")
    
    conn2 = graph.connect(print1.node_id, "exec_out", print2.node_id, "exec")
    print(f"   Connected: Print1 -> Print2 ({conn2})")
    
    # 4. Add a variable
    print("\n4. Adding variable...")
    counter = graph.add_variable("counter", "Integer", 0)
    print(f"   Added variable: {counter.name} = {counter.default_value}")
    
    # 5. Show graph stats
    print("\n5. Graph Statistics:")
    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Connections: {len(graph.connections)}")
    print(f"   Variables: {len(graph.variables)}")
    
    # 6. Find start nodes
    print("\n6. Finding entry points...")
    entry_points = graph.find_start_nodes()
    for node in entry_points:
        print(f"   Entry point: {node}")
    
    # 7. Serialize to JSON
    print("\n7. Serializing graph...")
    data = graph.to_dict()
    json_str = json.dumps(data, indent=2)
    print(f"   Serialized to {len(json_str)} characters of JSON")
    
    # Show a snippet
    print("\n   JSON Preview (first 500 chars):")
    print("   " + json_str[:500].replace("\n", "\n   ") + "...")
    
    # 8. Test error handling
    print("\n8. Testing error handling...")
    print1.set_error("Test error message")
    print(f"   Set error on Print1: has_error={print1.has_error}")
    print(f"   Error message: {print1.error_message}")
    print1.clear_error()
    print(f"   Cleared error: has_error={print1.has_error}")
    
    # 9. Remove a node
    print("\n9. Removing node...")
    print(f"   Before: {len(graph.nodes)} nodes, {len(graph.connections)} connections")
    graph.remove_node(print2.node_id)
    print(f"   After removing Print2: {len(graph.nodes)} nodes, {len(graph.connections)} connections")
    
    print("\n" + "=" * 60)
    print("Demo Complete! Phase 1 Core Node System is working.")
    print("=" * 60)


if __name__ == "__main__":
    main()
