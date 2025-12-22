# -*- coding: utf-8 -*-
"""
GraphExecutor - Engine for executing node graphs.

Provides:
- Execution flow traversal (following execution pins)
- Data flow evaluation (pulling from connected outputs)
- Variable management
- Error handling with node location
- Async execution support
"""
import asyncio
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from loguru import logger

if TYPE_CHECKING:
    from ..core.graph import NodeGraph
    from ..core.base_node import BaseNode
    from ..core.pins import BasePin


class ExecutionState(Enum):
    """Current state of graph execution."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class ExecutionLog:
    """A single log entry during execution."""
    timestamp: float
    node_id: str
    node_type: str
    message: str
    level: str = "INFO"  # INFO, DEBUG, WARNING, ERROR
    pin_name: Optional[str] = None


@dataclass
class ExecutionContext:
    """
    Context for graph execution.
    
    Holds runtime state including:
    - Variable values
    - Execution log
    - Current node being executed
    - Error state
    
    Attributes:
        variables: Dict of variable name -> current value
        logs: List of execution log entries
        current_node_id: ID of node currently executing
        error: Error message if execution failed
        error_node_id: ID of node where error occurred
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    logs: List[ExecutionLog] = field(default_factory=list)
    current_node_id: Optional[str] = None
    error: Optional[str] = None
    error_node_id: Optional[str] = None
    error_pin_name: Optional[str] = None
    
    # Internal state for loops
    _loop_state: Dict[str, Any] = field(default_factory=dict)
    _gate_state: Dict[str, bool] = field(default_factory=dict)
    _do_once_state: Dict[str, bool] = field(default_factory=dict)
    _flipflop_state: Dict[str, bool] = field(default_factory=dict)
    
    def log(self, node: 'BaseNode', message: str, level: str = "INFO", pin_name: str = None):
        """Add a log entry."""
        import time
        entry = ExecutionLog(
            timestamp=time.time(),
            node_id=node.node_id,
            node_type=node.node_type,
            message=message,
            level=level,
            pin_name=pin_name
        )
        self.logs.append(entry)
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{node.node_type}] {message}")
    
    def set_error(self, node: 'BaseNode', error: str, pin_name: str = None):
        """Set error state."""
        self.error = error
        self.error_node_id = node.node_id
        self.error_pin_name = pin_name
        node.set_error(error)
        self.log(node, f"ERROR: {error}", "ERROR", pin_name)
    
    def get_variable(self, name: str) -> Any:
        """Get variable value."""
        return self.variables.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set variable value."""
        self.variables[name] = value


class GraphExecutor:
    """
    Engine for executing node graphs.
    
    Executes graphs by following execution pins from start nodes,
    evaluating data connections on demand, and managing variables.
    
    Features:
    - Start from any entry point (Start, Event nodes)
    - Follow execution flow through connected exec pins
    - Evaluate data pins by pulling from connected outputs
    - Handle loops, branches, and other flow control
    - Capture errors with node/pin location
    
    Signals (for Qt integration):
        on_node_started: Emitted when node begins execution
        on_node_completed: Emitted when node finishes
        on_variable_changed: Emitted when variable value changes
        on_error: Emitted on execution error
        on_log: Emitted for each log entry
    """
    
    def __init__(self, graph: 'NodeGraph'):
        """
        Create an executor for a graph.
        
        Args:
            graph: The NodeGraph to execute
        """
        self.graph = graph
        self.state = ExecutionState.IDLE
        self.context = ExecutionContext()
        
        # Callbacks
        self._on_node_started: List[Callable] = []
        self._on_node_completed: List[Callable] = []
        self._on_variable_changed: List[Callable] = []
        self._on_error: List[Callable] = []
        self._on_log: List[Callable] = []
        
        # Debug
        self._breakpoints: set = set()  # Node IDs
        self._step_mode = False
        self._step_event: Optional[asyncio.Event] = None
    
    # =========================================================================
    # Execution Control
    # =========================================================================
    
    def run(self) -> ExecutionContext:
        """
        Run the graph synchronously.
        
        Returns:
            ExecutionContext with results
        """
        return asyncio.get_event_loop().run_until_complete(self.run_async())
    
    async def run_async(self) -> ExecutionContext:
        """
        Run the graph asynchronously.
        
        Returns:
            ExecutionContext with results
        """
        self.reset()
        self.state = ExecutionState.RUNNING
        
        # Initialize variables from graph
        for name, var in self.graph.variables.items():
            self.context.variables[name] = var.default_value
        
        # Find entry points
        start_nodes = self.graph.find_start_nodes()
        
        if not start_nodes:
            self.context.error = "No entry point found (Start node required)"
            self.state = ExecutionState.ERROR
            return self.context
        
        # Execute from each start node
        for start_node in start_nodes:
            if self.state == ExecutionState.ERROR:
                break
            await self._execute_from_node(start_node)
        
        if self.state != ExecutionState.ERROR:
            self.state = ExecutionState.COMPLETED
        
        return self.context
    
    def reset(self):
        """Reset execution state."""
        self.state = ExecutionState.IDLE
        self.context = ExecutionContext()
        
        # Clear node errors
        for node in self.graph.nodes.values():
            node.clear_error()
    
    def pause(self):
        """Pause execution."""
        if self.state == ExecutionState.RUNNING:
            self.state = ExecutionState.PAUSED
    
    def resume(self):
        """Resume paused execution."""
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.RUNNING
            if self._step_event:
                self._step_event.set()
    
    def step(self):
        """Execute single step."""
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.STEPPING
            if self._step_event:
                self._step_event.set()
    
    def stop(self):
        """Stop execution."""
        self.state = ExecutionState.IDLE
    
    # =========================================================================
    # Core Execution
    # =========================================================================
    
    async def _execute_from_node(self, node: 'BaseNode'):
        """
        Execute starting from a node, following execution flow.
        
        Args:
            node: Node to start from
        """
        await self._execute_node(node)
        
        # Follow execution output pins
        for pin_name, pin in node.output_pins.items():
            if pin.pin_type.name == "EXECUTION" and pin._connections:
                for conn in pin._connections:
                    next_node = conn.target_pin.node
                    if next_node and self.state in (ExecutionState.RUNNING, ExecutionState.STEPPING):
                        await self._execute_from_node(next_node)
    
    async def _execute_node(self, node: 'BaseNode'):
        """
        Execute a single node.
        
        Args:
            node: Node to execute
        """
        # Check state
        if self.state not in (ExecutionState.RUNNING, ExecutionState.STEPPING):
            return
        
        # Breakpoint handling
        if node.node_id in self._breakpoints:
            self.state = ExecutionState.PAUSED
            self._step_event = asyncio.Event()
            await self._step_event.wait()
        
        # Update context
        self.context.current_node_id = node.node_id
        
        # Notify callbacks
        for cb in self._on_node_started:
            cb(node)
        
        try:
            # Get executor for this node type
            from .node_executor import get_executor
            executor = get_executor(node.node_type)
            
            if executor:
                # Execute with the specialized executor
                await executor.execute(node, self.context, self)
            else:
                # Default: just log that we executed
                self.context.log(node, f"Executed {node.node_type}")
            
        except Exception as e:
            self.context.set_error(node, str(e))
            self.state = ExecutionState.ERROR
            for cb in self._on_error:
                cb(node, str(e))
            return
        
        # Notify completion
        for cb in self._on_node_completed:
            cb(node)
        
        # Step mode - pause after each node
        if self._step_mode and self.state == ExecutionState.STEPPING:
            self.state = ExecutionState.PAUSED
    
    def evaluate_input(self, node: 'BaseNode', pin_name: str) -> Any:
        """
        Evaluate an input pin's value.
        
        If connected, pulls value from connected output.
        For data-only nodes (no exec pins), executes them first.
        Otherwise returns the default value.
        
        Args:
            node: Node containing the pin
            pin_name: Name of input pin
            
        Returns:
            Resolved value
        """
        pin = node.get_input_pin(pin_name)
        if not pin:
            return None
        
        if pin.connection:
            # Get value from connected output pin
            source_pin = pin.connection.source_pin
            source_node = source_pin.node
            
            # Check if source node is a data-only node (no exec input pins)
            if source_node:
                has_exec_input = any(
                    p.pin_type.name == "EXECUTION" 
                    for p in source_node.input_pins.values()
                )
                
                # If data-only node, execute it first (recursively)
                if not has_exec_input:
                    self._evaluate_data_node_sync(source_node)
            
            return source_pin.get_value()
        
        return pin.default_value
    
    def _evaluate_data_node_sync(self, node: 'BaseNode'):
        """
        Execute a data-only node synchronously.
        
        First evaluates all its inputs recursively, then executes the node.
        
        Args:
            node: Data node to evaluate
        """
        # First, recursively evaluate all inputs
        for pin in node.input_pins.values():
            if pin.connection:
                source_node = pin.connection.source_pin.node
                if source_node:
                    has_exec_input = any(
                        p.pin_type.name == "EXECUTION" 
                        for p in source_node.input_pins.values()
                    )
                    if not has_exec_input:
                        self._evaluate_data_node_sync(source_node)
        
        # Execute this node directly (sync path for data nodes)
        self._execute_data_node_directly(node)
    
    def _execute_data_node_directly(self, node: 'BaseNode'):
        """Execute a data node directly without async."""
        # For simple data nodes, we can evaluate inputs and set outputs directly
        # This is a simplified sync path for when we're already in an async context
        
        node_type = node.node_type
        
        # Handle common data-only nodes directly
        if node_type == "ToString":
            value = self.evaluate_input(node, "value")
            node.set_output("string", str(value) if value is not None else "")
        
        elif node_type == "ToInteger":
            value = self.evaluate_input(node, "value")
            try:
                node.set_output("integer", int(value) if value is not None else 0)
            except (ValueError, TypeError):
                node.set_output("integer", 0)
        
        elif node_type == "ToFloat":
            value = self.evaluate_input(node, "value")
            try:
                node.set_output("float", float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                node.set_output("float", 0.0)
        
        elif node_type == "ToBoolean":
            value = self.evaluate_input(node, "value")
            node.set_output("boolean", bool(value))
        
        elif node_type == "StringConcat":
            a = self.evaluate_input(node, "a") or ""
            b = self.evaluate_input(node, "b") or ""
            sep = self.evaluate_input(node, "separator") or ""
            node.set_output("result", f"{a}{sep}{b}")
        
        elif node_type == "StringLength":
            text = self.evaluate_input(node, "input") or ""
            node.set_output("length", len(text))
        
        elif node_type == "ArrayLength":
            array = self.evaluate_input(node, "array") or []
            node.set_output("length", len(array))
        
        elif node_type == "GetFileInfo":
            from pathlib import Path
            from datetime import datetime
            path = self.evaluate_input(node, "path") or ""
            try:
                p = Path(path)
                stat = p.stat()
                node.set_output("size", stat.st_size)
                node.set_output("modified", datetime.fromtimestamp(stat.st_mtime).isoformat())
                node.set_output("extension", p.suffix)
                node.set_output("filename", p.name)
            except Exception:
                node.set_output("size", 0)
                node.set_output("modified", "")
                node.set_output("extension", "")
                node.set_output("filename", "")
        
        elif node_type == "PathJoin":
            from pathlib import Path
            path1 = self.evaluate_input(node, "path1") or ""
            path2 = self.evaluate_input(node, "path2") or ""
            path3 = self.evaluate_input(node, "path3") or ""
            parts = [p for p in [path1, path2, path3] if p]
            result = str(Path(*parts)) if parts else ""
            node.set_output("result", result)
        
        elif node_type == "FileExists":
            from pathlib import Path
            path = self.evaluate_input(node, "path") or ""
            p = Path(path)
            node.set_output("exists", p.exists())
            node.set_output("is_file", p.is_file())
            node.set_output("is_directory", p.is_dir())
        
        elif node_type == "GetDirectory":
            from pathlib import Path
            path = self.evaluate_input(node, "path") or ""
            p = Path(path)
            node.set_output("directory", str(p.parent))
            node.set_output("filename", p.name)
        
        elif node_type == "ImageInfo":
            img = self.evaluate_input(node, "image")
            if img is not None:
                node.set_output("width", img.width)
                node.set_output("height", img.height)
                node.set_output("mode", img.mode)
                node.set_output("format", getattr(img, 'format', '') or "")
            else:
                node.set_output("width", 0)
                node.set_output("height", 0)
                node.set_output("mode", "")
                node.set_output("format", "")
        
        # Math nodes
        elif node_type == "Add":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 0
            node.set_output("result", float(a) + float(b))
        
        elif node_type == "Subtract":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 0
            node.set_output("result", float(a) - float(b))
        
        elif node_type == "Multiply":
            a = self.evaluate_input(node, "a") or 1
            b = self.evaluate_input(node, "b") or 1
            node.set_output("result", float(a) * float(b))
        
        elif node_type == "Divide":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 1
            node.set_output("result", float(a) / float(b) if b != 0 else 0)
        
        elif node_type == "Modulo":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 1
            node.set_output("result", float(a) % float(b) if b != 0 else 0)
        
        elif node_type == "Power":
            base = self.evaluate_input(node, "base") or 2
            exp = self.evaluate_input(node, "exponent") or 2
            node.set_output("result", float(base) ** float(exp))
        
        elif node_type == "SquareRoot":
            import math
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", math.sqrt(max(0, float(value))))
        
        elif node_type == "Absolute":
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", abs(float(value)))
        
        elif node_type == "Negate":
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", -float(value))
        
        elif node_type == "Sin":
            import math
            rad = self.evaluate_input(node, "radians") or 0
            node.set_output("result", math.sin(float(rad)))
        
        elif node_type == "Cos":
            import math
            rad = self.evaluate_input(node, "radians") or 0
            node.set_output("result", math.cos(float(rad)))
        
        elif node_type == "Tan":
            import math
            rad = self.evaluate_input(node, "radians") or 0
            node.set_output("result", math.tan(float(rad)))
        
        elif node_type == "Asin":
            import math
            value = max(-1, min(1, self.evaluate_input(node, "value") or 0))
            node.set_output("radians", math.asin(float(value)))
        
        elif node_type == "Acos":
            import math
            value = max(-1, min(1, self.evaluate_input(node, "value") or 0))
            node.set_output("radians", math.acos(float(value)))
        
        elif node_type == "Atan":
            import math
            value = self.evaluate_input(node, "value") or 0
            node.set_output("radians", math.atan(float(value)))
        
        elif node_type == "Atan2":
            import math
            y = self.evaluate_input(node, "y") or 0
            x = self.evaluate_input(node, "x") or 1
            node.set_output("radians", math.atan2(float(y), float(x)))
        
        elif node_type == "DegreesToRadians":
            import math
            deg = self.evaluate_input(node, "degrees") or 0
            node.set_output("radians", math.radians(float(deg)))
        
        elif node_type == "RadiansToDegrees":
            import math
            rad = self.evaluate_input(node, "radians") or 0
            node.set_output("degrees", math.degrees(float(rad)))
        
        elif node_type == "Floor":
            import math
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", math.floor(float(value)))
        
        elif node_type == "Ceil":
            import math
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", math.ceil(float(value)))
        
        elif node_type == "Round":
            value = self.evaluate_input(node, "value") or 0
            decimals = self.evaluate_input(node, "decimals") or 0
            node.set_output("result", round(float(value), int(decimals)))
        
        elif node_type == "Min":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 0
            node.set_output("result", min(float(a), float(b)))
        
        elif node_type == "Max":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 0
            node.set_output("result", max(float(a), float(b)))
        
        elif node_type == "Clamp":
            value = self.evaluate_input(node, "value") or 0
            min_val = self.evaluate_input(node, "min") or 0
            max_val = self.evaluate_input(node, "max") or 1
            node.set_output("result", max(float(min_val), min(float(max_val), float(value))))
        
        elif node_type == "Lerp":
            a = self.evaluate_input(node, "a") or 0
            b = self.evaluate_input(node, "b") or 1
            t = self.evaluate_input(node, "t") or 0.5
            node.set_output("result", float(a) + (float(b) - float(a)) * float(t))
        
        elif node_type == "MapRange":
            value = self.evaluate_input(node, "value") or 0
            in_min = self.evaluate_input(node, "in_min") or 0
            in_max = self.evaluate_input(node, "in_max") or 1
            out_min = self.evaluate_input(node, "out_min") or 0
            out_max = self.evaluate_input(node, "out_max") or 1
            if in_max != in_min:
                t = (float(value) - float(in_min)) / (float(in_max) - float(in_min))
                result = float(out_min) + t * (float(out_max) - float(out_min))
            else:
                result = float(out_min)
            node.set_output("result", result)
        
        elif node_type == "Random":
            import random
            node.set_output("value", random.random())
        
        elif node_type == "RandomRange":
            import random
            min_val = self.evaluate_input(node, "min") or 0
            max_val = self.evaluate_input(node, "max") or 1
            node.set_output("value", random.uniform(float(min_val), float(max_val)))
        
        elif node_type == "RandomInt":
            import random
            min_val = self.evaluate_input(node, "min") or 0
            max_val = self.evaluate_input(node, "max") or 100
            node.set_output("value", random.randint(int(min_val), int(max_val)))
        
        elif node_type == "Log":
            import math
            value = self.evaluate_input(node, "value") or 1
            node.set_output("result", math.log(max(0.0001, float(value))))
        
        elif node_type == "Log10":
            import math
            value = self.evaluate_input(node, "value") or 1
            node.set_output("result", math.log10(max(0.0001, float(value))))
        
        elif node_type == "Exp":
            import math
            value = self.evaluate_input(node, "value") or 0
            node.set_output("result", math.exp(float(value)))
        
        elif node_type == "Pi":
            import math
            node.set_output("pi", math.pi)
        
        elif node_type == "E":
            import math
            node.set_output("e", math.e)
        
        # Comparison nodes
        elif node_type == "Equal":
            a = self.evaluate_input(node, "a")
            b = self.evaluate_input(node, "b")
            node.set_output("result", a == b)
        
        elif node_type == "NotEqual":
            a = self.evaluate_input(node, "a")
            b = self.evaluate_input(node, "b")
            node.set_output("result", a != b)
        
        elif node_type == "Greater":
            a = float(self.evaluate_input(node, "a") or 0)
            b = float(self.evaluate_input(node, "b") or 0)
            node.set_output("result", a > b)
        
        elif node_type == "Less":
            a = float(self.evaluate_input(node, "a") or 0)
            b = float(self.evaluate_input(node, "b") or 0)
            node.set_output("result", a < b)
        
        elif node_type == "GreaterEqual":
            a = float(self.evaluate_input(node, "a") or 0)
            b = float(self.evaluate_input(node, "b") or 0)
            node.set_output("result", a >= b)
        
        elif node_type == "LessEqual":
            a = float(self.evaluate_input(node, "a") or 0)
            b = float(self.evaluate_input(node, "b") or 0)
            node.set_output("result", a <= b)
    
    # =========================================================================
    # Flow Control Helpers
    # =========================================================================
    
    async def execute_output_pin(self, node: 'BaseNode', pin_name: str):
        """
        Execute nodes connected to a specific output execution pin.
        
        Args:
            node: Node containing the output pin
            pin_name: Name of output pin
        """
        pin = node.get_output_pin(pin_name)
        if not pin or pin.pin_type.name != "EXECUTION":
            return
        
        for conn in pin._connections:
            next_node = conn.target_pin.node
            if next_node:
                await self._execute_from_node(next_node)
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def on_node_started(self, callback: Callable):
        """Register callback for node start."""
        self._on_node_started.append(callback)
    
    def on_node_completed(self, callback: Callable):
        """Register callback for node completion."""
        self._on_node_completed.append(callback)
    
    def on_variable_changed(self, callback: Callable):
        """Register callback for variable change."""
        self._on_variable_changed.append(callback)
    
    def on_error(self, callback: Callable):
        """Register callback for errors."""
        self._on_error.append(callback)
    
    def add_breakpoint(self, node_id: str):
        """Add breakpoint at node."""
        self._breakpoints.add(node_id)
    
    def remove_breakpoint(self, node_id: str):
        """Remove breakpoint from node."""
        self._breakpoints.discard(node_id)
