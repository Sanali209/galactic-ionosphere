# -*- coding: utf-8 -*-
"""
Pins - Node input/output pin system with type checking.

Pin Types:
    EXECUTION - Flow control (white)
    BOOLEAN - True/False (red)
    INTEGER - Whole numbers (blue)
    FLOAT - Decimal numbers (green)
    STRING - Text (magenta)
    OBJECT - Generic objects (cyan)
    ARRAY - Lists (orange)
    IMAGE - Image data (gold)
    PATH - File paths (brown)
    ANY - Wildcard type (gray)

Example:
    # Execution pin for flow control
    exec_pin = ExecutionPin("exec")
    
    # Data pin with default value
    value_pin = DataPin("count", PinType.INTEGER, default_value=10)
"""
from enum import Enum, auto
from typing import Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .base_node import BaseNode
    from .connection import NodeConnection


class PinDirection(Enum):
    """Direction of a pin - input or output."""
    INPUT = auto()
    OUTPUT = auto()


class PinType(Enum):
    """
    Pin data types with color coding for UI.
    
    Each type is a tuple of (display_name, hex_color).
    """
    EXECUTION = ("Execution", "#FFFFFF")
    BOOLEAN = ("Boolean", "#CC0000")
    INTEGER = ("Integer", "#1E90FF")
    FLOAT = ("Float", "#00FF00")
    STRING = ("String", "#FF00FF")
    OBJECT = ("Object", "#00FFFF")
    ARRAY = ("Array", "#FFA500")
    IMAGE = ("Image", "#FFD700")
    PATH = ("Path", "#8B4513")
    ANY = ("Any", "#808080")
    
    @property
    def display_name(self) -> str:
        """Human-readable name for UI."""
        return self.value[0]
    
    @property
    def color(self) -> str:
        """Hex color for pin rendering."""
        return self.value[1]


@dataclass
class BasePin:
    """
    Base class for all pins.
    
    Attributes:
        name: Unique name within the node
        pin_type: Data type of this pin
        direction: INPUT or OUTPUT
        default_value: Default value when not connected (for inputs)
        node: Parent node reference
        connection: Connected NodeConnection (for input pins)
        cached_value: Last computed value (for output pins)
    """
    name: str
    pin_type: PinType
    direction: PinDirection = PinDirection.INPUT
    default_value: Any = None
    node: Optional['BaseNode'] = field(default=None, repr=False)
    connection: Optional['NodeConnection'] = field(default=None, repr=False)
    
    # For output pins - stores computed value
    cached_value: Any = field(default=None, repr=False)
    
    # For multiple connections from output pins
    _connections: List['NodeConnection'] = field(default_factory=list, repr=False)
    
    def can_connect_to(self, other: 'BasePin') -> bool:
        """
        Check if connection to another pin is valid.
        
        Rules:
        1. Must be different directions (input <-> output)
        2. Types must be compatible (same type or ANY wildcard)
        
        Args:
            other: Target pin to check
            
        Returns:
            True if connection is allowed
        """
        # Different directions required
        if self.direction == other.direction:
            return False
        
        # Execution pins can only connect to execution pins
        if self.pin_type == PinType.EXECUTION or other.pin_type == PinType.EXECUTION:
            return self.pin_type == other.pin_type
        
        # ANY type is wildcard
        if self.pin_type == PinType.ANY or other.pin_type == PinType.ANY:
            return True
        
        return self.pin_type == other.pin_type
    
    def get_value(self) -> Any:
        """
        Get current value of this pin.
        
        For output pins: Returns cached_value
        For input pins: Returns value from connection or default_value
        
        Returns:
            Current pin value
        """
        if self.direction == PinDirection.OUTPUT:
            return self.cached_value
        elif self.connection:
            return self.connection.source_pin.get_value()
        return self.default_value
    
    def is_connected(self) -> bool:
        """Check if this pin has any connections."""
        if self.direction == PinDirection.INPUT:
            return self.connection is not None
        else:
            return len(self._connections) > 0
    
    def __hash__(self) -> int:
        # Use name and node_id for hashing
        node_id = self.node.node_id if self.node else ""
        return hash((self.name, node_id))


class ExecutionPin(BasePin):
    """
    Special pin for execution flow.
    
    Execution pins control the order of node execution.
    They are always white colored and can only connect
    to other execution pins.
    """
    def __init__(self, name: str, direction: PinDirection = PinDirection.INPUT):
        """
        Create an execution pin.
        
        Args:
            name: Pin name (e.g., "exec", "then", "completed")
            direction: INPUT or OUTPUT
        """
        super().__init__(
            name=name,
            pin_type=PinType.EXECUTION,
            direction=direction
        )


class DataPin(BasePin):
    """
    Pin for data flow.
    
    Data pins carry values between nodes. They have a specific
    type and can have default values that users can edit
    in the Properties Panel when not connected.
    """
    def __init__(
        self, 
        name: str, 
        pin_type: PinType,
        direction: PinDirection = PinDirection.INPUT,
        default_value: Any = None
    ):
        """
        Create a data pin.
        
        Args:
            name: Pin name (e.g., "value", "count", "path")
            pin_type: Data type (INTEGER, STRING, etc.)
            direction: INPUT or OUTPUT
            default_value: Default value when not connected
        """
        if pin_type == PinType.EXECUTION:
            raise ValueError("Use ExecutionPin for execution flow")
        
        super().__init__(
            name=name,
            pin_type=pin_type,
            direction=direction,
            default_value=default_value
        )
