# -*- coding: utf-8 -*-
"""
String Nodes - String manipulation operations.

Provides nodes for:
- Concatenation
- Split/Join
- Replace
- Format with placeholders
- Length, Contains checks
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class StringConcatNode(BaseNode):
    """Concatenate two strings."""
    node_type = "StringConcat"
    metadata = NodeMetadata(
        category="String",
        display_name="Concat",
        description="Concatenate two strings",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("b", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("separator", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class StringSplitNode(BaseNode):
    """Split string by delimiter."""
    node_type = "StringSplit"
    metadata = NodeMetadata(
        category="String",
        display_name="Split",
        description="Split string by delimiter",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("delimiter", PinType.STRING, default_value=","))
        self.add_output_pin(DataPin("parts", PinType.ARRAY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("count", PinType.INTEGER, PinDirection.OUTPUT))


class StringReplaceNode(BaseNode):
    """Replace substring in string."""
    node_type = "StringReplace"
    metadata = NodeMetadata(
        category="String",
        display_name="Replace",
        description="Replace substring",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("search", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("replace", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class StringFormatNode(BaseNode):
    """Format string with placeholders {0}, {1}, etc."""
    node_type = "StringFormat"
    metadata = NodeMetadata(
        category="String",
        display_name="Format",
        description="Format string with {0}, {1} placeholders",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("template", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("arg0", PinType.ANY))
        self.add_input_pin(DataPin("arg1", PinType.ANY))
        self.add_input_pin(DataPin("arg2", PinType.ANY))
        self.add_input_pin(DataPin("arg3", PinType.ANY))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class StringLengthNode(BaseNode):
    """Get length of string."""
    node_type = "StringLength"
    metadata = NodeMetadata(
        category="String",
        display_name="Length",
        description="Get string length",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("length", PinType.INTEGER, PinDirection.OUTPUT))


class StringContainsNode(BaseNode):
    """Check if string contains substring."""
    node_type = "StringContains"
    metadata = NodeMetadata(
        category="String",
        display_name="Contains",
        description="Check if string contains substring",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("search", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("case_sensitive", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(DataPin("contains", PinType.BOOLEAN, PinDirection.OUTPUT))


class StringStartsWithNode(BaseNode):
    """Check if string starts with prefix."""
    node_type = "StringStartsWith"
    metadata = NodeMetadata(
        category="String",
        display_name="Starts With",
        description="Check if string starts with prefix",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("prefix", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.BOOLEAN, PinDirection.OUTPUT))


class StringEndsWithNode(BaseNode):
    """Check if string ends with suffix."""
    node_type = "StringEndsWith"
    metadata = NodeMetadata(
        category="String",
        display_name="Ends With",
        description="Check if string ends with suffix",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("suffix", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.BOOLEAN, PinDirection.OUTPUT))


class StringTrimNode(BaseNode):
    """Trim whitespace from string."""
    node_type = "StringTrim"
    metadata = NodeMetadata(
        category="String",
        display_name="Trim",
        description="Remove leading/trailing whitespace",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class StringUpperNode(BaseNode):
    """Convert to uppercase."""
    node_type = "StringUpper"
    metadata = NodeMetadata(
        category="String",
        display_name="To Upper",
        description="Convert to uppercase",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class StringLowerNode(BaseNode):
    """Convert to lowercase."""
    node_type = "StringLower"
    metadata = NodeMetadata(
        category="String",
        display_name="To Lower",
        description="Convert to lowercase",
        color="#FF00FF"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("input", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


# Export all nodes
ALL_NODES = [
    StringConcatNode,
    StringSplitNode,
    StringReplaceNode,
    StringFormatNode,
    StringLengthNode,
    StringContainsNode,
    StringStartsWithNode,
    StringEndsWithNode,
    StringTrimNode,
    StringUpperNode,
    StringLowerNode,
]
