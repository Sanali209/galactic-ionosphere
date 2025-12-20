# NodeGraph Developer Guide

Visual node programming subsystem for the Foundation template.

## Overview

The nodegraph module provides a complete visual programming environment with:
- 80 built-in nodes across 9 categories
- Execution engine with flow control
- Full PySide6 UI components
- Serialization for save/load

## Architecture

```
nodegraph/
├── core/           # Core data model
│   ├── base_node.py    # BaseNode class
│   ├── pins.py         # Pin types (Execution, Data)
│   ├── connection.py   # NodeConnection
│   ├── graph.py        # NodeGraph container
│   └── registry.py     # NodeRegistry
├── nodes/          # Built-in node definitions
│   ├── events.py       # Start, Update
│   ├── flow_control.py # Branch, Loop, Sequence
│   ├── variables.py    # Get/Set/Increment
│   ├── utilities.py    # Print, Type conversions
│   ├── file_nodes.py   # File I/O operations
│   ├── string_nodes.py # String manipulation
│   ├── array_nodes.py  # Array operations
│   ├── image_nodes.py  # PIL image processing
│   └── matplotlib_nodes.py # Chart visualization
├── execution/      # Execution engine
│   ├── executor.py     # GraphExecutor
│   └── node_executor.py # Node-specific executors
└── ui/             # PySide6 UI components
    ├── node_graph_widget.py  # Main canvas
    ├── node_item.py          # Node visual
    ├── pin_item.py           # Pin visual
    ├── connection_item.py    # Wire visual
    ├── properties_panel.py   # Property editor
    ├── node_palette_panel.py # Node browser
    ├── variables_panel.py    # Variable manager
    └── execution_log_panel.py # Log viewer
```

## Creating Custom Nodes

### 1. Define the Node Class

```python
from nodegraph.core import BaseNode, NodeMetadata
from nodegraph.core.pins import ExecutionPin, DataPin, PinType, PinDirection

class MyCustomNode(BaseNode):
    """My custom processing node."""
    
    node_type = "MyCustom"
    metadata = NodeMetadata(
        category="Custom",
        display_name="My Custom Node",
        description="Does something custom",
        color="#FF6600"
    )
    
    def _setup_pins(self):
        # Execution pins (for flow control)
        self.add_input_pin(ExecutionPin("exec"))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        
        # Data pins (for values)
        self.add_input_pin(DataPin("input", PinType.STRING))
        self.add_input_pin(DataPin("count", PinType.INTEGER, default_value=1))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))
```

### 2. Create the Executor

```python
from nodegraph.execution.node_executor import BaseNodeExecutor, register_executor

class MyCustomExecutor(BaseNodeExecutor):
    """Executor for MyCustom node."""
    
    async def execute(self, node, context, executor):
        # Get input values
        input_val = executor.evaluate_input(node, "input") or ""
        count = executor.evaluate_input(node, "count") or 1
        
        # Process
        result = input_val * count
        
        # Set output
        node.set_output("result", result)
        
        # Log (optional)
        context.log(node, f"Processed: {result}")
        
        # Continue execution flow
        await executor.execute_output_pin(node, "exec_out")

# Register the executor
register_executor("MyCustom", MyCustomExecutor())
```

### 3. Register the Node

```python
from nodegraph.core import NodeRegistry

registry = NodeRegistry()
registry.register(MyCustomNode)
```

## Pin Types

| Type | Python Type | Color |
|------|-------------|-------|
| EXECUTION | None | White |
| BOOLEAN | bool | Red |
| INTEGER | int | Cyan |
| FLOAT | float | Green |
| STRING | str | Magenta |
| ARRAY | list | Orange |
| OBJECT | Any | Gray |
| IMAGE | PIL.Image | Purple |

## Flow Control Nodes

### Branch
Executes True or False path based on condition.

### For Loop
Iterates from first_index to last_index, executing loop_body each iteration.

### For Each Loop
Iterates over array elements.

### While Loop
Repeats while condition is true.

### Sequence
Executes multiple outputs in order.

## Execution Model

1. **Start nodes** begin execution (no exec input pins)
2. **Execution pins** chain node order (white connections)
3. **Data pins** are evaluated on-demand when needed
4. **Variables** persist across the entire graph run

## Serialization

```python
# Save graph
data = graph.to_dict()
with open("my_graph.json", "w") as f:
    json.dump(data, f)

# Load graph
with open("my_graph.json") as f:
    data = json.load(f)
graph = NodeGraph.from_dict(data, registry)
```

## UI Integration

```python
from nodegraph.ui import NodeGraphWidget, PropertiesPanel

# Create widget
canvas = NodeGraphWidget()
canvas.set_graph(graph, registry)

# Connect signals
canvas.node_selected.connect(properties_panel.set_node)
canvas.graph_changed.connect(on_modified)
```
