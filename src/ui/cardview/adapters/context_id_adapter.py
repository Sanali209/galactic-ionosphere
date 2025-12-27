"""
ContextIdAdapter - Generates context IDs for state persistence.

Different adapters for folders, queries, and algorithm results.
"""
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class ContextIdAdapter(ABC):
    """
    Abstract base for context ID generation.
    
    Different adapters produce IDs from different sources,
    allowing state to be keyed by folder, query, or algorithm.
    
    Example:
        adapter = FolderContextAdapter()
        context_id = adapter.generate_id("/path/to/folder")
        # Returns: "folder:abc123..."
    """
    
    @abstractmethod
    def generate_id(self, source: Any) -> str:
        """
        Generate unique context ID from source.
        
        Args:
            source: Source data for ID generation
            
        Returns:
            Context ID string
        """
        pass


class FolderContextAdapter(ContextIdAdapter):
    """
    Generate context ID from folder path or ID.
    
    Example:
        adapter = FolderContextAdapter()
        context_id = adapter.generate_id("folder_objectid_123")
        # Returns: "folder:folder_objectid_123"
    """
    
    def generate_id(self, folder_id: str) -> str:
        """Generate ID from folder."""
        return f"folder:{folder_id}"


class QueryContextAdapter(ContextIdAdapter):
    """
    Generate context ID from database query.
    
    Hashes the query to create a consistent ID.
    
    Example:
        adapter = QueryContextAdapter()
        query = {"file_type": "image", "rating": {"$gte": 4}}
        context_id = adapter.generate_id(query)
        # Returns: "query:a1b2c3..."
    """
    
    def generate_id(self, query: Dict[str, Any]) -> str:
        """Generate ID from query hash."""
        # Sort keys for consistent hashing
        query_str = json.dumps(query, sort_keys=True, default=str)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()[:16]
        return f"query:{query_hash}"


class AlgorithmResultAdapter(ContextIdAdapter):
    """
    Generate context ID for algorithm results.
    
    Used to persist results of heavy computations
    (e.g., similarity sort, clustering).
    
    Example:
        adapter = AlgorithmResultAdapter()
        context_id = adapter.generate_id({
            "algorithm": "similarity_sort",
            "params": {"threshold": 0.8},
            "result_hash": "abc123"
        })
        # Returns: "algo:similarity_sort:abc123"
    """
    
    def generate_id(self, data: Dict[str, Any]) -> str:
        """
        Generate ID from algorithm info.
        
        Args:
            data: Dict with keys:
                - algorithm: Algorithm name
                - params: Algorithm parameters (optional)
                - result_hash: Hash of result order
        """
        algorithm = data.get("algorithm", "unknown")
        result_hash = data.get("result_hash", "")
        
        if not result_hash:
            # Generate hash from params
            params = data.get("params", {})
            params_str = json.dumps(params, sort_keys=True, default=str)
            result_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        return f"algo:{algorithm}:{result_hash}"


class CustomContextAdapter(ContextIdAdapter):
    """
    Custom context ID adapter with user-defined function.
    
    Example:
        adapter = CustomContextAdapter(lambda x: f"custom:{x['key']}")
    """
    
    def __init__(self, generator_fn):
        """
        Initialize with generator function.
        
        Args:
            generator_fn: Function(source) returning context ID string
        """
        self._generator = generator_fn
    
    def generate_id(self, source: Any) -> str:
        """Generate ID using custom function."""
        return self._generator(source)
