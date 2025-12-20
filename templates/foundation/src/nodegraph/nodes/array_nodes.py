# -*- coding: utf-8 -*-
"""
Array Nodes - Array/list manipulation.

Provides nodes for:
- Array creation and access
- Join/Split operations
- Filtering with wildcards
- Length, Get, Append, Merge
"""
import fnmatch
from typing import List, Any

from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class ArrayJoinNode(BaseNode):
    """Join array elements into string."""
    node_type = "ArrayJoin"
    metadata = NodeMetadata(
        category="Array",
        display_name="Join",
        description="Join array into string with separator",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("separator", PinType.STRING, default_value=","))
        self.add_output_pin(DataPin("result", PinType.STRING, PinDirection.OUTPUT))


class ArrayLengthNode(BaseNode):
    """Get array length."""
    node_type = "ArrayLength"
    metadata = NodeMetadata(
        category="Array",
        display_name="Length",
        description="Get array length",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_output_pin(DataPin("length", PinType.INTEGER, PinDirection.OUTPUT))


class ArrayGetNode(BaseNode):
    """Get element at index."""
    node_type = "ArrayGet"
    metadata = NodeMetadata(
        category="Array",
        display_name="Get",
        description="Get element at index",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("index", PinType.INTEGER, default_value=0))
        self.add_output_pin(DataPin("element", PinType.ANY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("valid", PinType.BOOLEAN, PinDirection.OUTPUT))


class ArraySetNode(BaseNode):
    """Set element at index."""
    node_type = "ArraySet"
    metadata = NodeMetadata(
        category="Array",
        display_name="Set",
        description="Set element at index",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("index", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("value", PinType.ANY))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArrayAppendNode(BaseNode):
    """Append element to array."""
    node_type = "ArrayAppend"
    metadata = NodeMetadata(
        category="Array",
        display_name="Append",
        description="Append element to array",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("element", PinType.ANY))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArrayMergeNode(BaseNode):
    """Merge two arrays."""
    node_type = "ArrayMerge"
    metadata = NodeMetadata(
        category="Array",
        display_name="Merge",
        description="Merge two arrays",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array1", PinType.ARRAY))
        self.add_input_pin(DataPin("array2", PinType.ARRAY))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArrayFilterNode(BaseNode):
    """Filter array with wildcard pattern."""
    node_type = "ArrayFilter"
    metadata = NodeMetadata(
        category="Array",
        display_name="Filter",
        description="Filter array with wildcard pattern (*.jpg;*.png)",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("pattern", PinType.STRING, default_value="*"))
        self.add_output_pin(DataPin("matched", PinType.ARRAY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("unmatched", PinType.ARRAY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("count", PinType.INTEGER, PinDirection.OUTPUT))


class ArrayReverseNode(BaseNode):
    """Reverse array order."""
    node_type = "ArrayReverse"
    metadata = NodeMetadata(
        category="Array",
        display_name="Reverse",
        description="Reverse array order",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArraySortNode(BaseNode):
    """Sort array."""
    node_type = "ArraySort"
    metadata = NodeMetadata(
        category="Array",
        display_name="Sort",
        description="Sort array",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("descending", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArrayUniqueNode(BaseNode):
    """Remove duplicates from array."""
    node_type = "ArrayUnique"
    metadata = NodeMetadata(
        category="Array",
        display_name="Unique",
        description="Remove duplicates",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


class ArraySliceNode(BaseNode):
    """Get slice of array."""
    node_type = "ArraySlice"
    metadata = NodeMetadata(
        category="Array",
        display_name="Slice",
        description="Get array slice [start:end]",
        color="#FFA500"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_input_pin(DataPin("start", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("end", PinType.INTEGER, default_value=-1))
        self.add_output_pin(DataPin("result", PinType.ARRAY, PinDirection.OUTPUT))


# Export all nodes
ALL_NODES = [
    ArrayJoinNode,
    ArrayLengthNode,
    ArrayGetNode,
    ArraySetNode,
    ArrayAppendNode,
    ArrayMergeNode,
    ArrayFilterNode,
    ArrayReverseNode,
    ArraySortNode,
    ArrayUniqueNode,
    ArraySliceNode,
]
