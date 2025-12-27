# -*- coding: utf-8 -*-
"""
Base Node - Abstract base class for all nodes.

Provides:
- Unique identification
- Pin management
- Metadata for display
- Serialization support

Example:
    class MyNode(BaseNode):
        node_type = "MyNode"
        metadata = NodeMetadata(category="Custom", display_name="My Node")
        
        def _setup_pins(self):
            self.add_input_pin(ExecutionPin("exec"))
            self.add_input_pin(DataPin("value", PinType.INTEGER, default_value=42))
            self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from uuid import uuid4

if TYPE_CHECKING:
    from .pins import BasePin, PinDirection


class NodeMetadata(BaseModel):
    """
    Node metadata for registry and display.
    
    Attributes:
        category: Category for grouping in palette (e.g., "Flow Control")
        display_name: Human-readable name shown in UI
        description: Tooltip description
        color: Hex color for node header (e.g., "#4A90D9")
        icon: Optional icon name or path
    """
    category: str = "General"
    display_name: str = ""
    description: str = ""
    color: str = "#4A90D9"
    icon: Optional[str] = None
    
    class Config:
        # Reason: Allow arbitrary types for flexibility
        arbitrary_types_allowed = True


class BaseNode(ABC):
    """
    Abstract base class for all nodes.
    
    All custom nodes must inherit from this class and implement:
    - node_type: Unique string identifier for serialization
    - metadata: NodeMetadata instance for display
    - _setup_pins(): Method to define input/output pins
    
    Attributes:
        node_id: Unique identifier for this node instance
        position: (x, y) canvas position tuple
    """
    node_type: str = "BaseNode"
    metadata: NodeMetadata = NodeMetadata()
    
    def __init__(self, node_id: Optional[str] = None):
        """
        Initialize a new node instance.
        
        Args:
            node_id: Optional unique ID (generated if not provided)
        """
        self.node_id = node_id or str(uuid4())
        self.position: tuple[float, float] = (0.0, 0.0)
        self._input_pins: Dict[str, 'BasePin'] = {}
        self._output_pins: Dict[str, 'BasePin'] = {}
        self._error_message: Optional[str] = None
        self._setup_pins()
    
    @abstractmethod
    def _setup_pins(self) -> None:
        """
        Define input/output pins. Must be implemented by subclasses.
        
        Example:
            def _setup_pins(self):
                self.add_input_pin(ExecutionPin("exec"))
                self.add_input_pin(DataPin("value", PinType.INTEGER))
                self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        """
        pass
    
    def add_input_pin(self, pin: 'BasePin') -> 'BasePin':
        """
        Add an input pin to this node.
        
        Args:
            pin: Pin instance to add
            
        Returns:
            The added pin (for chaining)
        """
        from .pins import PinDirection
        pin.node = self
        pin.direction = PinDirection.INPUT
        self._input_pins[pin.name] = pin
        return pin
    
    def add_output_pin(self, pin: 'BasePin') -> 'BasePin':
        """
        Add an output pin to this node.
        
        Args:
            pin: Pin instance to add
            
        Returns:
            The added pin (for chaining)
        """
        from .pins import PinDirection
        pin.node = self
        pin.direction = PinDirection.OUTPUT
        self._output_pins[pin.name] = pin
        return pin
    
    def get_input_pin(self, name: str) -> Optional['BasePin']:
        """Get an input pin by name."""
        return self._input_pins.get(name)
    
    def get_output_pin(self, name: str) -> Optional['BasePin']:
        """Get an output pin by name."""
        return self._output_pins.get(name)
    
    @property
    def input_pins(self) -> Dict[str, 'BasePin']:
        """Get all input pins."""
        return self._input_pins.copy()
    
    @property
    def output_pins(self) -> Dict[str, 'BasePin']:
        """Get all output pins."""
        return self._output_pins.copy()
    
    def get_input(self, name: str) -> Any:
        """
        Get value from an input pin.
        
        Resolves connections automatically - if the pin is connected,
        returns the value from the connected output pin. Otherwise,
        returns the pin's default_value.
        
        Args:
            name: Input pin name
            
        Returns:
            The resolved value
        """
        pin = self._input_pins.get(name)
        if pin is None:
            return None
        if pin.connection:
            return pin.connection.get_value()
        return pin.default_value
    
    def set_output(self, name: str, value: Any) -> None:
        """
        Set value on an output pin.
        
        Args:
            name: Output pin name
            value: Value to cache
        """
        pin = self._output_pins.get(name)
        if pin:
            pin.cached_value = value
    
    def set_error(self, message: str) -> None:
        """
        Set error state on this node.
        
        Args:
            message: Error message to display
        """
        self._error_message = message
    
    def clear_error(self) -> None:
        """Clear error state."""
        self._error_message = None
    
    @property
    def has_error(self) -> bool:
        """Check if node has an error."""
        return self._error_message is not None
    
    @property
    def error_message(self) -> Optional[str]:
        """Get current error message."""
        return self._error_message
    
    def to_dict(self) -> dict:
        """
        Serialize node for saving.
        
        Returns:
            Dictionary representation of node state
        """
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "position": list(self.position),
            "pin_values": {
                name: pin.default_value 
                for name, pin in self._input_pins.items()
                if pin.default_value is not None
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BaseNode':
        """
        Deserialize node from saved data.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            New node instance with restored state
        """
        node = cls(node_id=data.get("node_id"))
        pos = data.get("position", [0, 0])
        node.position = (float(pos[0]), float(pos[1]))
        
        # Restore pin values
        for name, value in data.get("pin_values", {}).items():
            if name in node._input_pins:
                node._input_pins[name].default_value = value
        
        return node
    
    def __repr__(self) -> str:
        return f"<{self.node_type}({self.node_id[:8]})>"
