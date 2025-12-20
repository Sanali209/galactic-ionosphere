# -*- coding: utf-8 -*-
"""
Event Nodes - Entry points for graph execution.

These nodes serve as starting points for execution flow,
similar to Unreal Engine's Event BeginPlay.
"""
from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class StartNode(BaseNode):
    """
    Entry point - execution begins here.
    
    The Start node is the primary entry point for a graph.
    Connect its exec output to begin the execution chain.
    """
    node_type = "Start"
    metadata = NodeMetadata(
        category="Events",
        display_name="Start",
        description="Entry point for graph execution (runs once)",
        color="#CC0000"
    )
    
    def _setup_pins(self):
        self.add_output_pin(ExecutionPin("exec", PinDirection.OUTPUT))


class UpdateNode(BaseNode):
    """
    Called every frame/tick.
    
    Similar to Unreal's Event Tick, this node fires
    repeatedly during execution (if enabled).
    """
    node_type = "Update"
    metadata = NodeMetadata(
        category="Events",
        display_name="Update (Tick)",
        description="Called every tick/frame",
        color="#CC0000"
    )
    
    def _setup_pins(self):
        self.add_output_pin(ExecutionPin("exec", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("delta_time", PinType.FLOAT, PinDirection.OUTPUT))


class CustomEventNode(BaseNode):
    """
    Custom named event that can be triggered.
    
    Create custom events that can be called from
    other parts of the graph using the Call Event node.
    """
    node_type = "CustomEvent"
    metadata = NodeMetadata(
        category="Events",
        display_name="Custom Event",
        description="Define a custom event",
        color="#CC0000"
    )
    
    def __init__(self, node_id=None, event_name: str = "MyEvent"):
        self.event_name = event_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_output_pin(ExecutionPin("exec", PinDirection.OUTPUT))
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["event_name"] = self.event_name
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CustomEventNode':
        node = cls(
            node_id=data.get("node_id"),
            event_name=data.get("event_name", "MyEvent")
        )
        pos = data.get("position", [0, 0])
        node.position = (float(pos[0]), float(pos[1]))
        return node


class CallEventNode(BaseNode):
    """
    Call a custom event by name.
    
    Triggers execution of a CustomEvent node
    with the matching event name.
    """
    node_type = "CallEvent"
    metadata = NodeMetadata(
        category="Events",
        display_name="Call Event",
        description="Trigger a custom event",
        color="#CC0000"
    )
    
    def __init__(self, node_id=None, event_name: str = "MyEvent"):
        self.event_name = event_name
        super().__init__(node_id)
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_output_pin(ExecutionPin("then", PinDirection.OUTPUT))
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["event_name"] = self.event_name
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CallEventNode':
        node = cls(
            node_id=data.get("node_id"),
            event_name=data.get("event_name", "MyEvent")
        )
        pos = data.get("position", [0, 0])
        node.position = (float(pos[0]), float(pos[1]))
        return node


# Export all nodes for registration
ALL_NODES = [
    StartNode,
    UpdateNode,
    CustomEventNode,
    CallEventNode,
]
