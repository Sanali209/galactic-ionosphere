# -*- coding: utf-8 -*-
"""
NodeGraph Nodes - Built-in node types.

Provides all built-in nodes organized by category:
- Events: Start, Update, CustomEvent
- Flow Control: If, Branch, Sequence, Loops
- Variables: Get, Set, Increment
- Utilities: Print, Type conversions
- Math: Arithmetic, Trig, Rounding, Random
- File: Read, Write, List, Copy, Move
- String: Concat, Split, Replace, Format
- Array: Join, Get, Filter, Merge
- Image: Load, Save, Resize, Crop
- Matplotlib: Figure, Line/Bar/Scatter plots
"""
from . import events
from . import flow_control
from . import variables
from . import utilities
from . import math_nodes
from . import file_nodes
from . import string_nodes
from . import array_nodes
from . import image_nodes
from . import matplotlib_nodes

# Collect all nodes for bulk registration
ALL_NODES = (
    events.ALL_NODES +
    flow_control.ALL_NODES +
    variables.ALL_NODES +
    utilities.ALL_NODES +
    math_nodes.MATH_NODES +
    file_nodes.ALL_NODES +
    string_nodes.ALL_NODES +
    array_nodes.ALL_NODES +
    image_nodes.ALL_NODES +
    matplotlib_nodes.ALL_NODES
)

