# -*- coding: utf-8 -*-
"""
Mathematical Nodes - Arithmetic and math function nodes.

Provides nodes for:
- Basic arithmetic (Add, Subtract, Multiply, Divide)
- Advanced math (Power, Sqrt, Abs, Mod)
- Trigonometry (Sin, Cos, Tan, Asin, Acos, Atan)
- Rounding (Floor, Ceil, Round)
- Comparison (Min, Max, Clamp)
- Random (Random, RandomRange)
"""
import math
import random
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import DataPin, PinType, PinDirection


# =============================================================================
# Basic Arithmetic
# =============================================================================

class AddNode(BaseNode):
    """Add two numbers."""
    
    node_type = "Add"
    metadata = NodeMetadata(
        category="Math",
        display_name="Add",
        description="Add two numbers (A + B)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class SubtractNode(BaseNode):
    """Subtract two numbers."""
    
    node_type = "Subtract"
    metadata = NodeMetadata(
        category="Math",
        display_name="Subtract",
        description="Subtract two numbers (A - B)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class MultiplyNode(BaseNode):
    """Multiply two numbers."""
    
    node_type = "Multiply"
    metadata = NodeMetadata(
        category="Math",
        display_name="Multiply",
        description="Multiply two numbers (A × B)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=1.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class DivideNode(BaseNode):
    """Divide two numbers."""
    
    node_type = "Divide"
    metadata = NodeMetadata(
        category="Math",
        display_name="Divide",
        description="Divide two numbers (A ÷ B)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=1.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class ModuloNode(BaseNode):
    """Modulo (remainder) of division."""
    
    node_type = "Modulo"
    metadata = NodeMetadata(
        category="Math",
        display_name="Modulo",
        description="Remainder after division (A mod B)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class PowerNode(BaseNode):
    """Raise to power."""
    
    node_type = "Power"
    metadata = NodeMetadata(
        category="Math",
        display_name="Power",
        description="Raise base to exponent (base^exp)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("base", PinType.FLOAT, default_value=2.0))
        self.add_input_pin(DataPin("exponent", PinType.FLOAT, default_value=2.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class SquareRootNode(BaseNode):
    """Square root."""
    
    node_type = "SquareRoot"
    metadata = NodeMetadata(
        category="Math",
        display_name="Square Root",
        description="Calculate square root",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=4.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class AbsoluteNode(BaseNode):
    """Absolute value."""
    
    node_type = "Absolute"
    metadata = NodeMetadata(
        category="Math",
        display_name="Absolute",
        description="Absolute value |x|",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class NegateNode(BaseNode):
    """Negate a number."""
    
    node_type = "Negate"
    metadata = NodeMetadata(
        category="Math",
        display_name="Negate",
        description="Negate value (-x)",
        color="#4CAF50"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Trigonometry
# =============================================================================

class SinNode(BaseNode):
    """Sine function."""
    
    node_type = "Sin"
    metadata = NodeMetadata(
        category="Math",
        display_name="Sin",
        description="Sine of angle (radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("radians", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class CosNode(BaseNode):
    """Cosine function."""
    
    node_type = "Cos"
    metadata = NodeMetadata(
        category="Math",
        display_name="Cos",
        description="Cosine of angle (radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("radians", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class TanNode(BaseNode):
    """Tangent function."""
    
    node_type = "Tan"
    metadata = NodeMetadata(
        category="Math",
        display_name="Tan",
        description="Tangent of angle (radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("radians", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class AsinNode(BaseNode):
    """Arc sine function."""
    
    node_type = "Asin"
    metadata = NodeMetadata(
        category="Math",
        display_name="Asin",
        description="Arc sine (returns radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("radians", PinType.FLOAT, PinDirection.OUTPUT))


class AcosNode(BaseNode):
    """Arc cosine function."""
    
    node_type = "Acos"
    metadata = NodeMetadata(
        category="Math",
        display_name="Acos",
        description="Arc cosine (returns radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("radians", PinType.FLOAT, PinDirection.OUTPUT))


class AtanNode(BaseNode):
    """Arc tangent function."""
    
    node_type = "Atan"
    metadata = NodeMetadata(
        category="Math",
        display_name="Atan",
        description="Arc tangent (returns radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("radians", PinType.FLOAT, PinDirection.OUTPUT))


class Atan2Node(BaseNode):
    """Arc tangent of y/x."""
    
    node_type = "Atan2"
    metadata = NodeMetadata(
        category="Math",
        display_name="Atan2",
        description="Arc tangent of y/x (returns radians)",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("y", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("x", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("radians", PinType.FLOAT, PinDirection.OUTPUT))


class DegreesToRadiansNode(BaseNode):
    """Convert degrees to radians."""
    
    node_type = "DegreesToRadians"
    metadata = NodeMetadata(
        category="Math",
        display_name="Degrees to Radians",
        description="Convert degrees to radians",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("degrees", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("radians", PinType.FLOAT, PinDirection.OUTPUT))


class RadiansToDegreesNode(BaseNode):
    """Convert radians to degrees."""
    
    node_type = "RadiansToDegrees"
    metadata = NodeMetadata(
        category="Math",
        display_name="Radians to Degrees",
        description="Convert radians to degrees",
        color="#2196F3"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("radians", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("degrees", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Rounding
# =============================================================================

class FloorNode(BaseNode):
    """Floor function."""
    
    node_type = "Floor"
    metadata = NodeMetadata(
        category="Math",
        display_name="Floor",
        description="Round down to nearest integer",
        color="#9C27B0"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.INTEGER, PinDirection.OUTPUT))


class CeilNode(BaseNode):
    """Ceiling function."""
    
    node_type = "Ceil"
    metadata = NodeMetadata(
        category="Math",
        display_name="Ceil",
        description="Round up to nearest integer",
        color="#9C27B0"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.INTEGER, PinDirection.OUTPUT))


class RoundNode(BaseNode):
    """Round to nearest."""
    
    node_type = "Round"
    metadata = NodeMetadata(
        category="Math",
        display_name="Round",
        description="Round to nearest integer or decimal places",
        color="#9C27B0"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("decimals", PinType.INTEGER, default_value=0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Comparison / Range
# =============================================================================

class MinNode(BaseNode):
    """Minimum of two values."""
    
    node_type = "Min"
    metadata = NodeMetadata(
        category="Math",
        display_name="Min",
        description="Return the smaller of two values",
        color="#FF9800"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class MaxNode(BaseNode):
    """Maximum of two values."""
    
    node_type = "Max"
    metadata = NodeMetadata(
        category="Math",
        display_name="Max",
        description="Return the larger of two values",
        color="#FF9800"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class ClampNode(BaseNode):
    """Clamp value to range."""
    
    node_type = "Clamp"
    metadata = NodeMetadata(
        category="Math",
        display_name="Clamp",
        description="Constrain value between min and max",
        color="#FF9800"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("min", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("max", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class LerpNode(BaseNode):
    """Linear interpolation."""
    
    node_type = "Lerp"
    metadata = NodeMetadata(
        category="Math",
        display_name="Lerp",
        description="Linear interpolation between A and B",
        color="#FF9800"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("a", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("b", PinType.FLOAT, default_value=1.0))
        self.add_input_pin(DataPin("t", PinType.FLOAT, default_value=0.5))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class MapRangeNode(BaseNode):
    """Map value from one range to another."""
    
    node_type = "MapRange"
    metadata = NodeMetadata(
        category="Math",
        display_name="Map Range",
        description="Remap value from input range to output range",
        color="#FF9800"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.5))
        self.add_input_pin(DataPin("in_min", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("in_max", PinType.FLOAT, default_value=1.0))
        self.add_input_pin(DataPin("out_min", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("out_max", PinType.FLOAT, default_value=100.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Random
# =============================================================================

class RandomNode(BaseNode):
    """Random float 0-1."""
    
    node_type = "Random"
    metadata = NodeMetadata(
        category="Math",
        display_name="Random",
        description="Random float between 0 and 1",
        color="#E91E63"
    )
    
    def _setup_pins(self):
        self.add_output_pin(DataPin("value", PinType.FLOAT, PinDirection.OUTPUT))


class RandomRangeNode(BaseNode):
    """Random float in range."""
    
    node_type = "RandomRange"
    metadata = NodeMetadata(
        category="Math",
        display_name="Random Range",
        description="Random float between min and max",
        color="#E91E63"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("min", PinType.FLOAT, default_value=0.0))
        self.add_input_pin(DataPin("max", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("value", PinType.FLOAT, PinDirection.OUTPUT))


class RandomIntNode(BaseNode):
    """Random integer in range."""
    
    node_type = "RandomInt"
    metadata = NodeMetadata(
        category="Math",
        display_name="Random Integer",
        description="Random integer between min and max (inclusive)",
        color="#E91E63"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("min", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("max", PinType.INTEGER, default_value=100))
        self.add_output_pin(DataPin("value", PinType.INTEGER, PinDirection.OUTPUT))


# =============================================================================
# Logarithmic / Exponential
# =============================================================================

class LogNode(BaseNode):
    """Natural logarithm."""
    
    node_type = "Log"
    metadata = NodeMetadata(
        category="Math",
        display_name="Log",
        description="Natural logarithm (ln)",
        color="#795548"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class Log10Node(BaseNode):
    """Base-10 logarithm."""
    
    node_type = "Log10"
    metadata = NodeMetadata(
        category="Math",
        display_name="Log10",
        description="Base-10 logarithm",
        color="#795548"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


class ExpNode(BaseNode):
    """Exponential (e^x)."""
    
    node_type = "Exp"
    metadata = NodeMetadata(
        category="Math",
        display_name="Exp",
        description="Exponential function (e^x)",
        color="#795548"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("value", PinType.FLOAT, default_value=0.0))
        self.add_output_pin(DataPin("result", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Constants
# =============================================================================

class PiNode(BaseNode):
    """Pi constant."""
    
    node_type = "Pi"
    metadata = NodeMetadata(
        category="Math",
        display_name="Pi",
        description="π ≈ 3.14159...",
        color="#607D8B"
    )
    
    def _setup_pins(self):
        self.add_output_pin(DataPin("pi", PinType.FLOAT, PinDirection.OUTPUT))


class ENode(BaseNode):
    """Euler's number constant."""
    
    node_type = "E"
    metadata = NodeMetadata(
        category="Math",
        display_name="E",
        description="e ≈ 2.71828...",
        color="#607D8B"
    )
    
    def _setup_pins(self):
        self.add_output_pin(DataPin("e", PinType.FLOAT, PinDirection.OUTPUT))


# =============================================================================
# Export all math nodes
# =============================================================================

MATH_NODES = [
    # Basic arithmetic
    AddNode,
    SubtractNode,
    MultiplyNode,
    DivideNode,
    ModuloNode,
    PowerNode,
    SquareRootNode,
    AbsoluteNode,
    NegateNode,
    # Trigonometry
    SinNode,
    CosNode,
    TanNode,
    AsinNode,
    AcosNode,
    AtanNode,
    Atan2Node,
    DegreesToRadiansNode,
    RadiansToDegreesNode,
    # Rounding
    FloorNode,
    CeilNode,
    RoundNode,
    # Comparison/Range
    MinNode,
    MaxNode,
    ClampNode,
    LerpNode,
    MapRangeNode,
    # Random
    RandomNode,
    RandomRangeNode,
    RandomIntNode,
    # Logarithmic
    LogNode,
    Log10Node,
    ExpNode,
    # Constants
    PiNode,
    ENode,
]
