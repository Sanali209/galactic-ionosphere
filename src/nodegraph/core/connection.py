# -*- coding: utf-8 -*-
"""
Connection - Manages connections between node pins.

A connection links an output pin to an input pin, allowing
data or execution flow between nodes.

Example:
    # Create connection between nodes
    conn = NodeConnection(
        source_pin=node_a.get_output_pin("result"),
        target_pin=node_b.get_input_pin("value")
    )
"""
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from .pins import BasePin


class NodeConnection:
    """
    Represents a connection between two pins.
    
    Connections are directional: source (output) -> target (input).
    
    Attributes:
        connection_id: Unique identifier
        source_pin: Output pin (data/execution source)
        target_pin: Input pin (data/execution destination)
    """
    
    def __init__(
        self, 
        source_pin: 'BasePin', 
        target_pin: 'BasePin',
        connection_id: Optional[str] = None
    ):
        """
        Create a new connection between pins.
        
        Args:
            source_pin: Output pin (from node producing value)
            target_pin: Input pin (to node consuming value)
            connection_id: Optional ID (generated if not provided)
            
        Raises:
            ValueError: If connection is not valid (type mismatch, etc.)
        """
        self.connection_id = connection_id or str(uuid4())
        self.source_pin = source_pin
        self.target_pin = target_pin
        
        # Validate connection
        if not source_pin.can_connect_to(target_pin):
            raise ValueError(
                f"Cannot connect {source_pin.pin_type.display_name} "
                f"to {target_pin.pin_type.display_name}"
            )
        
        # Register connection on target (input pins have single connection)
        target_pin.connection = self
        
        # Register on source (output pins can have multiple connections)
        if self not in source_pin._connections:
            source_pin._connections.append(self)
    
    def disconnect(self) -> None:
        """
        Remove this connection from both pins.
        
        Cleans up references on both source and target pins.
        """
        # Remove from target
        if self.target_pin.connection == self:
            self.target_pin.connection = None
        
        # Remove from source
        if self in self.source_pin._connections:
            self.source_pin._connections.remove(self)
    
    def get_value(self):
        """
        Get value from source pin.
        
        Returns:
            The cached value from the source output pin
        """
        return self.source_pin.get_value()
    
    def to_dict(self) -> dict:
        """
        Serialize connection for saving.
        
        Returns:
            Dictionary representation of connection
        """
        return {
            "connection_id": self.connection_id,
            "source_node_id": self.source_pin.node.node_id,
            "source_pin_name": self.source_pin.name,
            "target_node_id": self.target_pin.node.node_id,
            "target_pin_name": self.target_pin.name,
        }
    
    def __repr__(self) -> str:
        src_node = self.source_pin.node.node_id[:8] if self.source_pin.node else "?"
        tgt_node = self.target_pin.node.node_id[:8] if self.target_pin.node else "?"
        return (
            f"<Connection {src_node}.{self.source_pin.name} -> "
            f"{tgt_node}.{self.target_pin.name}>"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, NodeConnection):
            return False
        return self.connection_id == other.connection_id
    
    def __hash__(self) -> int:
        return hash(self.connection_id)
