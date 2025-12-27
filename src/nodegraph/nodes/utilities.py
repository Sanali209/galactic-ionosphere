# -*- coding: utf-8 -*-
"""
Utility Nodes - Common helper nodes.

Provides general-purpose utility nodes for debugging,
logging, and data manipulation.
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class PrintNode(BaseNode):
    """
    Print message to console.
    
    Useful for debugging - outputs the message
    to the execution log.
    """
    node_type = "Print"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="Print",
        description="Print message to console",
        color="#808080"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("message", PinType.STRING, default_value=""))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))


class PrintVariableNode(BaseNode):
    """
    Print a variable with its name.
    
    Outputs "name: value" format for debugging.
    """
    node_type = "PrintVariable"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="Print Variable",
        description="Print variable name and value",
        color="#808080"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("name", PinType.STRING, default_value="var"))
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(ExecutionPin("exec_out", PinDirection.OUTPUT))


class CommentNode(BaseNode):
    """
    Comment node - for documentation.
    
    Does nothing during execution, just displays
    a comment in the graph editor.
    """
    node_type = "Comment"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="Comment",
        description="Add a comment to the graph",
        color="#FFFF99"
    )
    
    def __init__(self, node_id=None, comment: str = "Add your comment here"):
        self.comment = comment
        super().__init__(node_id)
    
    def _setup_pins(self):
        # No pins - comment only
        pass
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["comment"] = self.comment
        return data


class MakeArrayNode(BaseNode):
    """
    Create an array from individual values.
    
    Combines multiple inputs into a single array output.
    """
    node_type = "MakeArray"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="Make Array",
        description="Create array from values",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("item_0", PinType.ANY))
        self.add_input_pin(DataPin("item_1", PinType.ANY))
        self.add_input_pin(DataPin("item_2", PinType.ANY))
        self.add_input_pin(DataPin("item_3", PinType.ANY))
        self.add_output_pin(DataPin("array", PinType.ARRAY, PinDirection.OUTPUT))


class ToStringNode(BaseNode):
    """
    Convert any value to string.
    
    Useful for displaying non-string values in Print nodes.
    """
    node_type = "ToString"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="To String",
        description="Convert value to string",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("string", PinType.STRING, PinDirection.OUTPUT))


class ToIntegerNode(BaseNode):
    """
    Convert value to integer.
    """
    node_type = "ToInteger"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="To Integer",
        description="Convert value to integer",
        color="#1E90FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("integer", PinType.INTEGER, PinDirection.OUTPUT))


class ToFloatNode(BaseNode):
    """
    Convert value to float.
    """
    node_type = "ToFloat"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="To Float",
        description="Convert value to float",
        color="#00FF00"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("float", PinType.FLOAT, PinDirection.OUTPUT))


class ToBooleanNode(BaseNode):
    """
    Convert value to boolean.
    
    Standard Python truthiness rules apply.
    """
    node_type = "ToBoolean"
    metadata = NodeMetadata(
        category="Utilities",
        display_name="To Boolean",
        description="Convert value to boolean",
        color="#CC0000"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("boolean", PinType.BOOLEAN, PinDirection.OUTPUT))


# Export all nodes for registration
ALL_NODES = [
    PrintNode,
    PrintVariableNode,
    CommentNode,
    MakeArrayNode,
    ToStringNode,
    ToIntegerNode,
    ToFloatNode,
    ToBooleanNode,
]
