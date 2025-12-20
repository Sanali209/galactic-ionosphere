# NodeGraph API Reference

## Core Classes

### BaseNode
Base class for all nodes.

```python
class BaseNode:
    node_type: str          # Unique type identifier
    metadata: NodeMetadata  # Display info
    node_id: str           # Instance UUID
    position: Tuple[float, float]
    input_pins: Dict[str, BasePin]
    output_pins: Dict[str, BasePin]
    
    def _setup_pins(self) -> None
    def add_input_pin(pin: BasePin) -> None
    def add_output_pin(pin: BasePin) -> None
    def get_input_pin(name: str) -> Optional[BasePin]
    def get_output_pin(name: str) -> Optional[BasePin]
    def set_output(name: str, value: Any) -> None
    def to_dict() -> dict
    def from_dict(data: dict) -> BaseNode
```

### NodeMetadata
Node display information.

```python
@dataclass
class NodeMetadata:
    category: str = "Uncategorized"
    display_name: str = ""
    description: str = ""
    color: str = "#4a5568"
    icon: str = ""
```

### NodeGraph
Container for nodes and connections.

```python
class NodeGraph:
    name: str
    nodes: Dict[str, BaseNode]
    connections: Dict[str, NodeConnection]
    variables: Dict[str, Variable]
    
    def add_node(node: BaseNode) -> BaseNode
    def remove_node(node_id: str) -> None
    def get_node(node_id: str) -> Optional[BaseNode]
    def connect(src_node, src_pin, tgt_node, tgt_pin) -> NodeConnection
    def disconnect(connection_id: str) -> None
    def add_variable(name, type, default) -> Variable
    def find_start_nodes() -> List[BaseNode]
    def to_dict() -> dict
    def from_dict(data, registry) -> NodeGraph
```

### NodeRegistry
Central node type registry.

```python
class NodeRegistry:
    def register(node_cls: Type[BaseNode]) -> None
    def unregister(node_type: str) -> None
    def get_node_class(node_type: str) -> Optional[Type[BaseNode]]
    def get_all_nodes() -> List[Type[BaseNode]]
    def get_categories() -> Dict[str, List[Type[BaseNode]]]
    def create_node(node_type: str) -> Optional[BaseNode]
    def search_nodes(query: str) -> List[Type[BaseNode]]
```

## Pins

### ExecutionPin
Flow control pin.

```python
ExecutionPin(name: str, direction: PinDirection = INPUT)
```

### DataPin
Value transfer pin.

```python
DataPin(
    name: str,
    pin_type: PinType,
    direction: PinDirection = INPUT,
    default_value: Any = None
)
```

### PinType Enum
```python
class PinType(Enum):
    EXECUTION = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    ARRAY = auto()
    OBJECT = auto()
    IMAGE = auto()
```

## Execution

### GraphExecutor
Runs node graphs.

```python
class GraphExecutor:
    def __init__(graph: NodeGraph)
    
    async def run_async() -> ExecutionContext
    def run() -> ExecutionContext
    def reset() -> None
    def pause() -> None
    def resume() -> None
    def stop() -> None
    
    def evaluate_input(node, pin_name) -> Any
    async def execute_output_pin(node, pin_name) -> None
```

### ExecutionContext
Runtime state.

```python
@dataclass
class ExecutionContext:
    variables: Dict[str, Any]
    logs: List[ExecutionLog]
    error: Optional[str]
    error_node_id: Optional[str]
    
    def log(node, message, level="INFO")
    def set_error(node, message)
    def get_variable(name) -> Any
    def set_variable(name, value)
```

### BaseNodeExecutor
Base for node executors.

```python
class BaseNodeExecutor:
    async def execute(node, context, executor) -> None
```

## UI Components

### NodeGraphWidget
Main canvas.

```python
class NodeGraphWidget(QGraphicsView):
    # Signals
    node_selected = Signal(object)
    connection_created = Signal(object)
    graph_changed = Signal()
    
    def set_graph(graph, registry)
    def add_node(node_type, position) -> NodeItem
    def delete_selected()
    def copy_selected()
    def paste()
    def fit_in_view()
```

### PropertiesPanel
Node property editor.

```python
class PropertiesPanel(QWidget):
    def set_node(node: Optional[BaseNode])
```

### NodePalettePanel
Node browser.

```python
class NodePalettePanel(QWidget):
    node_requested = Signal(str)  # node_type
    
    def set_registry(registry)
```

### VariablesPanel
Variable manager.

```python
class VariablesPanel(QWidget):
    def set_graph(graph)
```

### ExecutionLogPanel
Log viewer.

```python
class ExecutionLogPanel(QWidget):
    node_clicked = Signal(str)  # node_id
    
    def clear()
    def add_log(log: ExecutionLog)
    def add_message(message, level)
```
