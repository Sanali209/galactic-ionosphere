# -*- coding: utf-8 -*-
"""
NodeGraph - Visual Node Programming Subsystem

A visual node-based programming environment inspired by Unreal Engine Blueprints.
Provides execution flow, data connections, variables, and custom node creation.
"""

from .core.base_node import BaseNode, NodeMetadata
from .core.pins import (
    PinDirection,
    PinType,
    BasePin,
    ExecutionPin,
    DataPin,
)
from .core.connection import NodeConnection
from .core.graph import NodeGraph
from .core.registry import NodeRegistry

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseNode",
    "NodeMetadata",
    "PinDirection",
    "PinType",
    "BasePin",
    "ExecutionPin",
    "DataPin",
    "NodeConnection",
    "NodeGraph",
    "NodeRegistry",
]
