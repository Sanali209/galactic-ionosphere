# Core Modules

## UCoreFS (Universal Core Filesystem)

UCoreFS is more than just a file tracker; it's a semantic data engine.

### Features
- **File Tracking**: Monitors file changes, moves, and deletions.
- **Metadata Extraction**: Automatically extracts metadata from images, text, and other supported formats.
- **AI Embeddings**: Generates vector embeddings for content, enabling semantic search (e.g., "Find images that look like a sunset").
- **Hybrid Search**: Combine keyword search with vector similarity.

### Key Components
- `FSService`: The main entry point for file operations.
- `IndexerService`: Background service that processes files and generates embeddings.
- `DatabaseManager`: Handles connections to MongoDB (metadata) and ChromaDB (vectors).

## NodeGraph

The NodeGraph module provides a robust backend and UI for visual programming.

### Architecture
- **Graph**: The container for nodes and links.
- **Node**: A functional unit with Inputs and Outputs (Pins).
- **Pin**: Connection points for data flow.
- **Executor**: The engine that traverses the graph and executes nodes.

### Creating a Custom Node

```python
from nodegraph.core.node import Node
from nodegraph.core.pins import InputPin, OutputPin

class AddNode(Node):
    def __init__(self):
        super().__init__("Add")
        self.add_pin(InputPin("A", "float"))
        self.add_pin(InputPin("B", "float"))
        self.add_pin(OutputPin("Result", "float"))

    def execute(self, context):
        a = self.get_input_value("A")
        b = self.get_input_value("B")
        self.set_output_value("Result", a + b)
```
