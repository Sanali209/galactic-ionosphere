# Node Editor Sample Application

A complete visual node programming environment built with PySide6.

## Features

- **Visual Node Editor** - Drag-and-drop node creation and connection
- **80 Built-in Nodes** - Flow control, variables, file I/O, strings, arrays, images, charts
- **Execution Engine** - Run graphs with real-time logging
- **File Operations** - Save/load graphs as JSON
- **Dark Theme** - Modern UI with dockable panels

## Quick Start

```bash
# From the foundation template directory
cd templates/foundation

# Run the node editor
python samples/node_editor/main.py
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Graph | Ctrl+N |
| Open | Ctrl+O |
| Save | Ctrl+S |
| Copy | Ctrl+C |
| Cut | Ctrl+X |
| Paste | Ctrl+V |
| Duplicate | Ctrl+D |
| Delete | Delete |
| Select All | Ctrl+A |
| Fit All | F |
| Execute | F5 |
| Stop | Escape |

## Canvas Navigation

- **Pan**: Middle-mouse drag or Space+Left drag
- **Zoom**: Mouse wheel
- **Select**: Click nodes, Ctrl+Click for multi-select
- **Box Select**: Left-drag on empty space

## Panels

### Node Palette (Left)
Browse available nodes by category. Double-click or drag to add nodes.

### Variables (Left, tabbed)
Manage graph-level variables. Add Integer, Float, String, Boolean, or Array variables.

### Properties (Right)
Edit properties of the selected node. Auto-generates widgets based on pin types.

### Execution Log (Bottom)
View execution trace with timestamps, color-coded log levels, and error details.

## Example Graphs

Located in `examples/`:

- **hello_world.graph** - Simple Start → Print chain
- **counter_loop.graph** - Variable with ForLoop counting
- **file_filter.graph** - List directory and filter by pattern

## Node Categories

| Category | Nodes | Description |
|----------|-------|-------------|
| Events | 2 | Start, Update |
| Flow Control | 11 | If, Branch, Sequence, Loops, etc. |
| Variables | 4 | Get, Set, Increment, IsValid |
| Utilities | 8 | Print, Type conversions, MakeArray |
| File | 11 | Read, Write, List, Copy, Move, Delete |
| String | 11 | Concat, Split, Replace, Format |
| Array | 11 | Join, Get, Filter, Sort, Merge |
| Image | 9 | Load, Save, Resize, Crop, Rotate |
| Matplotlib | 11 | Figure, Line, Bar, Scatter, Pie |

## Creating Custom Nodes

```python
from nodegraph.core import BaseNode, NodeMetadata
from nodegraph.core.pins import ExecutionPin, DataPin, PinType, PinDirection

class MyCustomNode(BaseNode):
    node_type = "MyCustom"
    metadata = NodeMetadata(
        category="Custom",
        display_name="My Node",
        description="Does something custom",
        color="#FF6600"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("input", PinType.STRING))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("output", PinType.STRING, PinDirection.OUTPUT))
```

Then register it:

```python
registry.register(MyCustomNode)
```

And create an executor:

```python
from nodegraph.execution.node_executor import BaseNodeExecutor, register_executor

class MyCustomExecutor(BaseNodeExecutor):
    async def execute(self, node, context, executor):
        input_val = executor.evaluate_input(node, "input") or ""
        node.set_output("output", input_val.upper())
        await executor.execute_output_pin(node, "exec_out")

register_executor("MyCustom", MyCustomExecutor())
```

## Architecture

```
samples/node_editor/
├── main.py              # Main application entry point
└── examples/            # Example graph files
    ├── hello_world.graph
    ├── counter_loop.graph
    └── file_filter.graph

src/nodegraph/
├── core/                # Core data model
│   ├── base_node.py     # BaseNode class
│   ├── pins.py          # Pin types
│   ├── connection.py    # Connection class
│   ├── graph.py         # NodeGraph class
│   └── registry.py      # NodeRegistry
├── nodes/               # Built-in nodes
│   ├── events.py
│   ├── flow_control.py
│   ├── variables.py
│   ├── utilities.py
│   ├── file_nodes.py
│   ├── string_nodes.py
│   ├── array_nodes.py
│   ├── image_nodes.py
│   └── matplotlib_nodes.py
├── execution/           # Execution engine
│   ├── executor.py
│   └── node_executor.py
└── ui/                  # UI components
    ├── node_graph_widget.py
    ├── node_item.py
    ├── pin_item.py
    ├── connection_item.py
    ├── properties_panel.py
    ├── node_palette_panel.py
    ├── variables_panel.py
    └── execution_log_panel.py
```

## License

Part of the Foundation Template - MIT License
