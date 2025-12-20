# -*- coding: utf-8 -*-
"""
Variable Nodes - Get and set graph variables.

Variables are graph-level storage that persist across
execution and can be accessed from any node.
"""
from typing import Any
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class GetVariableNode(BaseNode):
    """
    Get value of a variable.
    
    Reads the current value of a graph variable.
    Pure node (no execution pins) - evaluated on demand.
    """
    node_type = "GetVariable"
    metadata = NodeMetadata(
        category="Variables",
        display_name="Get Variable",
        description="Read value from a variable",
        color="#00CC66"
    )
    
    def __init__(self, node_id=None, variable_name: str = "MyVar"):
        """
        Create Get Variable node.
        
        Args:
            node_id: Optional node ID
            variable_name: Name of variable to read
        """
        self.variable_name = variable_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_output_pin(DataPin("value", PinType.ANY, PinDirection.OUTPUT))
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["variable_name"] = self.variable_name
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GetVariableNode':
        node = cls(
            node_id=data.get("node_id"),
            variable_name=data.get("variable_name", "MyVar")
        )
        pos = data.get("position", [0, 0])
        node.position = (float(pos[0]), float(pos[1]))
        return node


class SetVariableNode(BaseNode):
    """
    Set value of a variable.
    
    Writes a new value to a graph variable.
    Has execution pins for flow control.
    """
    node_type = "SetVariable"
    metadata = NodeMetadata(
        category="Variables",
        display_name="Set Variable",
        description="Write value to a variable",
        color="#00CC66"
    )
    
    def __init__(self, node_id=None, variable_name: str = "MyVar"):
        """
        Create Set Variable node.
        
        Args:
            node_id: Optional node ID
            variable_name: Name of variable to write
        """
        self.variable_name = variable_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("new_value", PinType.ANY, PinDirection.OUTPUT))
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["variable_name"] = self.variable_name
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SetVariableNode':
        node = cls(
            node_id=data.get("node_id"),
            variable_name=data.get("variable_name", "MyVar")
        )
        pos = data.get("position", [0, 0])
        node.position = (float(pos[0]), float(pos[1]))
        for name, value in data.get("pin_values", {}).items():
            if name in node._input_pins:
                node._input_pins[name].default_value = value
        return node


class IncrementVariableNode(BaseNode):
    """
    Increment a numeric variable.
    
    Adds a value to the variable (default: 1).
    Convenience node for counters.
    """
    node_type = "IncrementVariable"
    metadata = NodeMetadata(
        category="Variables",
        display_name="Increment Variable",
        description="Add to a numeric variable",
        color="#00CC66"
    )
    
    def __init__(self, node_id=None, variable_name: str = "Counter"):
        self.variable_name = variable_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("amount", PinType.INTEGER, default_value=1))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("new_value", PinType.INTEGER, PinDirection.OUTPUT))
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["variable_name"] = self.variable_name
        return data


class IsValidNode(BaseNode):
    """
    Check if variable is valid (not None).
    
    Useful for checking if a variable has been set
    before using it.
    """
    node_type = "IsValid"
    metadata = NodeMetadata(
        category="Variables",
        display_name="Is Valid",
        description="Check if value is not None",
        color="#00CC66"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("is_valid", PinType.BOOLEAN, PinDirection.OUTPUT))


# Export all nodes for registration
ALL_NODES = [
    GetVariableNode,
    SetVariableNode,
    IncrementVariableNode,
    IsValidNode,
]
