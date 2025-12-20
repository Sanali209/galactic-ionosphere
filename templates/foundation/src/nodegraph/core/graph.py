# -*- coding: utf-8 -*-
"""
NodeGraph - Container for nodes and connections.

A NodeGraph represents a complete visual program that can be
executed, saved, and loaded.

Example:
    graph = NodeGraph("My Workflow")
    
    # Add nodes
    start = graph.add_node(StartNode())
    print_node = graph.add_node(PrintNode())
    
    # Connect them
    graph.connect(
        start.node_id, "exec",
        print_node.node_id, "exec"
    )
    
    # Save
    data = graph.to_dict()
"""
from typing import Dict, List, Optional, Type, Any, TYPE_CHECKING
from loguru import logger

from .base_node import BaseNode
from .connection import NodeConnection

if TYPE_CHECKING:
    from .registry import NodeRegistry


class Variable:
    """
    Graph-level variable.
    
    Variables can be accessed across nodes using Get/Set Variable nodes.
    
    Attributes:
        name: Variable name
        var_type: Type hint string
        default_value: Initial value
        value: Current runtime value
    """
    
    def __init__(
        self, 
        name: str, 
        var_type: str = "Any",
        default_value: Any = None
    ):
        self.name = name
        self.var_type = var_type
        self.default_value = default_value
        self.value = default_value
    
    def to_dict(self) -> dict:
        """Serialize variable."""
        return {
            "name": self.name,
            "var_type": self.var_type,
            "default_value": self.default_value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Variable':
        """Deserialize variable."""
        return cls(
            name=data["name"],
            var_type=data.get("var_type", "Any"),
            default_value=data.get("default_value")
        )


class NodeGraph:
    """
    Container for nodes and connections.
    
    Represents a complete visual program that can be
    executed, saved, and loaded.
    
    Attributes:
        name: Human-readable graph name
        nodes: Dictionary of node_id -> BaseNode
        connections: Dictionary of connection_id -> NodeConnection
        variables: Dictionary of name -> Variable
    """
    
    def __init__(self, name: str = "Untitled Graph"):
        """
        Create a new node graph.
        
        Args:
            name: Human-readable name for this graph
        """
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: Dict[str, NodeConnection] = {}
        self._variables: Dict[str, Variable] = {}
    
    # =========================================================================
    # Node Management
    # =========================================================================
    
    def add_node(self, node: BaseNode) -> BaseNode:
        """
        Add a node to the graph.
        
        Args:
            node: Node instance to add
            
        Returns:
            The added node (for chaining)
        """
        self.nodes[node.node_id] = node
        logger.debug(f"Added node: {node}")
        return node
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its connections.
        
        Args:
            node_id: ID of node to remove
        """
        node = self.nodes.get(node_id)
        if not node:
            return
        
        # Remove all connections to/from this node
        connections_to_remove = [
            conn_id for conn_id, conn in self.connections.items()
            if (conn.source_pin.node and conn.source_pin.node.node_id == node_id) or
               (conn.target_pin.node and conn.target_pin.node.node_id == node_id)
        ]
        for conn_id in connections_to_remove:
            self.disconnect(conn_id)
        
        del self.nodes[node_id]
        logger.debug(f"Removed node: {node}")
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def clear(self) -> None:
        """Remove all nodes and connections."""
        # Disconnect all
        for conn_id in list(self.connections.keys()):
            self.disconnect(conn_id)
        self.nodes.clear()
        self._variables.clear()
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    def connect(
        self, 
        source_node_id: str, 
        source_pin_name: str,
        target_node_id: str, 
        target_pin_name: str
    ) -> NodeConnection:
        """
        Create a connection between two nodes.
        
        Args:
            source_node_id: ID of node with output pin
            source_pin_name: Name of output pin
            target_node_id: ID of node with input pin
            target_pin_name: Name of input pin
            
        Returns:
            The created connection
            
        Raises:
            KeyError: If node or pin not found
            ValueError: If connection is invalid
        """
        source_node = self.nodes.get(source_node_id)
        target_node = self.nodes.get(target_node_id)
        
        if not source_node:
            raise KeyError(f"Source node not found: {source_node_id}")
        if not target_node:
            raise KeyError(f"Target node not found: {target_node_id}")
        
        source_pin = source_node.get_output_pin(source_pin_name)
        target_pin = target_node.get_input_pin(target_pin_name)
        
        if not source_pin:
            raise KeyError(f"Source pin not found: {source_pin_name}")
        if not target_pin:
            raise KeyError(f"Target pin not found: {target_pin_name}")
        
        # Remove existing connection on target input pin
        if target_pin.connection:
            self.disconnect(target_pin.connection.connection_id)
        
        # Create new connection
        connection = NodeConnection(source_pin, target_pin)
        self.connections[connection.connection_id] = connection
        
        logger.debug(f"Connected: {connection}")
        return connection
    
    def disconnect(self, connection_id: str) -> None:
        """
        Remove a connection.
        
        Args:
            connection_id: ID of connection to remove
        """
        conn = self.connections.get(connection_id)
        if conn:
            conn.disconnect()
            del self.connections[connection_id]
            logger.debug(f"Disconnected: {conn}")
    
    def get_connections_from_node(self, node_id: str) -> List[NodeConnection]:
        """Get all connections originating from a node."""
        return [
            conn for conn in self.connections.values()
            if conn.source_pin.node and conn.source_pin.node.node_id == node_id
        ]
    
    def get_connections_to_node(self, node_id: str) -> List[NodeConnection]:
        """Get all connections going into a node."""
        return [
            conn for conn in self.connections.values()
            if conn.target_pin.node and conn.target_pin.node.node_id == node_id
        ]
    
    # =========================================================================
    # Variable Management
    # =========================================================================
    
    def add_variable(
        self, 
        name: str, 
        var_type: str = "Any",
        default_value: Any = None
    ) -> Variable:
        """
        Add a graph-level variable.
        
        Args:
            name: Variable name
            var_type: Type hint string
            default_value: Initial value
            
        Returns:
            The created Variable
        """
        var = Variable(name, var_type, default_value)
        self._variables[name] = var
        return var
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        return self._variables.get(name)
    
    def remove_variable(self, name: str) -> None:
        """Remove a variable."""
        if name in self._variables:
            del self._variables[name]
    
    @property
    def variables(self) -> Dict[str, Variable]:
        """Get all variables."""
        return self._variables.copy()
    
    # =========================================================================
    # Execution Helpers
    # =========================================================================
    
    def find_start_nodes(self) -> List[BaseNode]:
        """
        Find all entry point nodes (Start, Event nodes).
        
        Returns:
            List of nodes that can begin execution
        """
        start_nodes = []
        for node in self.nodes.values():
            # Check if it's a start/event node (has exec output but no exec input)
            has_exec_out = any(
                pin.pin_type.name == "EXECUTION" 
                for pin in node.output_pins.values()
            )
            has_exec_in = any(
                pin.pin_type.name == "EXECUTION" 
                for pin in node.input_pins.values()
            )
            if has_exec_out and not has_exec_in:
                start_nodes.append(node)
        
        return start_nodes
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> dict:
        """
        Serialize graph for saving.
        
        Returns:
            Dictionary representation of entire graph
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections.values()],
            "variables": [var.to_dict() for var in self._variables.values()]
        }
    
    @classmethod
    def from_dict(cls, data: dict, registry: 'NodeRegistry') -> 'NodeGraph':
        """
        Load graph from saved data.
        
        Args:
            data: Dictionary from to_dict()
            registry: NodeRegistry for creating node instances
            
        Returns:
            Restored NodeGraph instance
        """
        graph = cls(name=data.get("name", "Untitled"))
        
        # Restore variables first
        for var_data in data.get("variables", []):
            var = Variable.from_dict(var_data)
            graph._variables[var.name] = var
        
        # Restore nodes
        for node_data in data.get("nodes", []):
            node_type = node_data.get("node_type")
            node_cls = registry.get_node_class(node_type)
            if node_cls:
                node = node_cls.from_dict(node_data)
                graph.add_node(node)
            else:
                logger.warning(f"Unknown node type: {node_type}")
        
        # Restore connections
        for conn_data in data.get("connections", []):
            try:
                graph.connect(
                    conn_data["source_node_id"], 
                    conn_data["source_pin_name"],
                    conn_data["target_node_id"], 
                    conn_data["target_pin_name"]
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to restore connection: {e}")
        
        return graph
    
    def __repr__(self) -> str:
        return f"<NodeGraph '{self.name}' nodes={len(self.nodes)} conn={len(self.connections)}>"
