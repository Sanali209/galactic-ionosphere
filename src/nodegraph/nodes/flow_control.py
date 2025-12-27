# -*- coding: utf-8 -*-
"""
Flow Control Nodes - Branching and looping constructs.

Provides nodes for controlling execution flow:
- If/Branch for conditional execution
- Sequence for ordered multi-output execution
- Loops (For, ForEach, While)
- Utility nodes (DoOnce, Gate, FlipFlop)
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class IfNode(BaseNode):
    """
    Simple If-Then (executes only if condition is true).
    
    Unlike Branch, this only has a single output that
    fires when condition is true. Useful for simple guards.
    """
    node_type = "If"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="If",
        description="Execute only if condition is true",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("condition", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("then", PinDirection.OUTPUT))


class BranchNode(BaseNode):
    """
    Conditional branching (if/else).
    
    Executes either the True or False output based on
    the condition input. Classic if/else construct.
    """
    node_type = "Branch"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Branch (If/Else)",
        description="Execute True or False path based on condition",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("condition", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("true", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("false", PinDirection.OUTPUT))


class SequenceNode(BaseNode):
    """
    Execute multiple outputs in order.
    
    All outputs fire sequentially, one after another.
    Useful for organizing execution flow.
    """
    node_type = "Sequence"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Sequence",
        description="Execute multiple outputs sequentially",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_output_pin(ExecutionPin("then_0", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("then_1", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("then_2", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("then_3", PinDirection.OUTPUT))


class ForLoopNode(BaseNode):
    """
    For loop with index.
    
    Loops from first_index to last_index (inclusive),
    executing loop_body each iteration with current index.
    """
    node_type = "ForLoop"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="For Loop",
        description="Loop from First to Last index",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("first_index", PinType.INTEGER, default_value=0))
        self.add_input_pin(DataPin("last_index", PinType.INTEGER, default_value=10))
        self.add_output_pin(ExecutionPin("loop_body", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("index", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("completed", PinDirection.OUTPUT))


class ForEachLoopNode(BaseNode):
    """
    Iterate over array.
    
    Executes loop_body for each element in the array,
    providing both the element and its index.
    """
    node_type = "ForEachLoop"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="For Each Loop",
        description="Loop over each element in array",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("array", PinType.ARRAY))
        self.add_output_pin(ExecutionPin("loop_body", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("element", PinType.ANY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("index", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("completed", PinDirection.OUTPUT))


class WhileLoopNode(BaseNode):
    """
    While loop.
    
    Continues executing loop_body as long as
    condition remains true. Be careful of infinite loops!
    """
    node_type = "WhileLoop"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="While Loop",
        description="Loop while condition is true",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("condition", PinType.BOOLEAN, default_value=True))
        self.add_output_pin(ExecutionPin("loop_body", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("completed", PinDirection.OUTPUT))


class DoOnceNode(BaseNode):
    """
    Execute only once until reset.
    
    The first time triggered, executes the output.
    Subsequent triggers are ignored until Reset is triggered.
    """
    node_type = "DoOnce"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Do Once",
        description="Execute output only once until reset",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(ExecutionPin("reset"))
        self.add_input_pin(DataPin("start_closed", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("completed", PinDirection.OUTPUT))


class GateNode(BaseNode):
    """
    Gate that can be opened/closed.
    
    When open, Enter passes through to Exit.
    When closed, Enter is blocked. Toggle switches state.
    """
    node_type = "Gate"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Gate",
        description="Control passage of execution",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("enter"))
        self.add_input_pin(ExecutionPin("open"))
        self.add_input_pin(ExecutionPin("close"))
        self.add_input_pin(ExecutionPin("toggle"))
        self.add_input_pin(DataPin("start_closed", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("exit", PinDirection.OUTPUT))


class FlipFlopNode(BaseNode):
    """
    Alternates between two outputs.
    
    Each trigger alternates between A and B outputs.
    Also provides is_A boolean output.
    """
    node_type = "FlipFlop"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Flip Flop",
        description="Alternate between A and B outputs",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_output_pin(ExecutionPin("A", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("B", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("is_A", PinType.BOOLEAN, PinDirection.OUTPUT))


class BreakLoopNode(BaseNode):
    """
    Break out of current loop.
    
    When executed inside a loop, immediately jumps to
    the loop's Completed output.
    """
    node_type = "BreakLoop"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Break",
        description="Exit the current loop",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))


class DelayNode(BaseNode):
    """
    Delay execution.
    
    Waits for specified duration before continuing.
    Non-blocking (uses async execution).
    """
    node_type = "Delay"
    metadata = NodeMetadata(
        category="Flow Control",
        display_name="Delay",
        description="Wait for specified duration",
        color="#4A90D9"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("duration", PinType.FLOAT, default_value=1.0))
        self.add_output_pin(ExecutionPin("completed", PinDirection.OUTPUT))


# Export all nodes for registration
ALL_NODES = [
    IfNode,
    BranchNode,
    SequenceNode,
    ForLoopNode,
    ForEachLoopNode,
    WhileLoopNode,
    DoOnceNode,
    GateNode,
    FlipFlopNode,
    BreakLoopNode,
    DelayNode,
]
