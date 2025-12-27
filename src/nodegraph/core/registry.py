# -*- coding: utf-8 -*-
"""
NodeRegistry - Central registry for all available node types.

The registry integrates with Foundation's ServiceLocator pattern
and provides node discovery, registration, and creation.

Example:
    # Get registry from service locator
    registry = sl.get_system(NodeRegistry)
    
    # Register custom node
    registry.register(MyCustomNode)
    
    # Create node instance
    node = registry.create_node("MyCustomNode")
"""
from typing import Type, Dict, List, Optional
from loguru import logger

from src.core.base_system import BaseSystem
from .base_node import BaseNode, NodeMetadata


class NodeRegistry(BaseSystem):
    """
    Central registry for all available node types.
    
    Allows dynamic node registration and discovery.
    Integrates with Foundation's ServiceLocator.
    
    Attributes:
        _node_classes: Dict mapping node_type -> node class
        _categories: Dict mapping category -> list of node classes
    """
    
    def __init__(self, locator=None, config=None):
        """
        Initialize the node registry.
        
        Args:
            locator: ServiceLocator instance (optional for standalone mode)
            config: ConfigManager instance (optional for standalone mode)
        """
        # Support standalone mode without ServiceLocator
        if locator is not None:
            super().__init__(locator, config)
        else:
            self._locator = None
            self._config = None
            self._initialized = False
        
        self._node_classes: Dict[str, Type[BaseNode]] = {}
        self._categories: Dict[str, List[Type[BaseNode]]] = {}
    
    def register_all_builtin(self) -> None:
        """Register all built-in nodes (for standalone mode)."""
        self._register_builtin_nodes()
    
    async def initialize(self) -> None:
        """Load built-in nodes on startup."""
        logger.info("NodeRegistry initializing...")
        self._register_builtin_nodes()
        if self._locator is not None:
            await super().initialize()
        self._initialized = True
        logger.info(f"NodeRegistry ready with {len(self._node_classes)} node types")
    
    async def shutdown(self) -> None:
        """Clean up on shutdown."""
        self._node_classes.clear()
        self._categories.clear()
        if self._locator is not None:
            await super().shutdown()
    
    def register(self, node_cls: Type[BaseNode]) -> None:
        """
        Register a node class.
        
        Args:
            node_cls: Node class to register (must have node_type attribute)
        """
        node_type = node_cls.node_type
        
        if node_type in self._node_classes:
            logger.warning(f"Overwriting existing node type: {node_type}")
        
        self._node_classes[node_type] = node_cls
        
        # Categorize by metadata.category
        category = node_cls.metadata.category
        if category not in self._categories:
            self._categories[category] = []
        
        if node_cls not in self._categories[category]:
            self._categories[category].append(node_cls)
        
        logger.debug(f"Registered node: {node_type} in category '{category}'")
    
    def unregister(self, node_type: str) -> None:
        """
        Unregister a node type.
        
        Args:
            node_type: Node type string to remove
        """
        if node_type in self._node_classes:
            node_cls = self._node_classes[node_type]
            del self._node_classes[node_type]
            
            # Remove from categories
            category = node_cls.metadata.category
            if category in self._categories:
                if node_cls in self._categories[category]:
                    self._categories[category].remove(node_cls)
    
    def get_node_class(self, node_type: str) -> Optional[Type[BaseNode]]:
        """
        Get node class by type name.
        
        Args:
            node_type: Node type string
            
        Returns:
            Node class or None if not found
        """
        return self._node_classes.get(node_type)
    
    def get_all_nodes(self) -> List[Type[BaseNode]]:
        """
        Get all registered node classes.
        
        Returns:
            List of all node classes
        """
        return list(self._node_classes.values())
    
    def get_categories(self) -> Dict[str, List[Type[BaseNode]]]:
        """
        Get nodes organized by category.
        
        Returns:
            Dict mapping category name -> list of node classes
        """
        return {k: list(v) for k, v in self._categories.items()}
    
    def create_node(self, node_type: str) -> Optional[BaseNode]:
        """
        Create a new instance of a node type.
        
        Args:
            node_type: Node type string
            
        Returns:
            New node instance or None if type not found
        """
        node_cls = self.get_node_class(node_type)
        if node_cls:
            return node_cls()
        return None
    
    def search_nodes(self, query: str) -> List[Type[BaseNode]]:
        """
        Search for nodes by name or description.
        
        Args:
            query: Search string (case-insensitive)
            
        Returns:
            List of matching node classes
        """
        query_lower = query.lower()
        results = []
        
        for node_cls in self._node_classes.values():
            meta = node_cls.metadata
            # Search in display_name, description, and node_type
            if (query_lower in meta.display_name.lower() or
                query_lower in meta.description.lower() or
                query_lower in node_cls.node_type.lower()):
                results.append(node_cls)
        
        return results
    
    def _register_builtin_nodes(self) -> None:
        """
        Register all built-in node types.
        
        Called during initialization to load standard nodes.
        """
        try:
            from ..nodes import ALL_NODES
            
            for node_cls in ALL_NODES:
                self.register(node_cls)
            
            logger.debug(f"Registered {len(ALL_NODES)} built-in nodes")
        except ImportError as e:
            logger.warning(f"Could not load built-in nodes: {e}")
