"""
UCoreFS - Query Builder

Fluent API for building complex MongoDB queries.
"""
from typing import Any, Dict, List, Optional
from bson import ObjectId


class Q:
    """Query expression builder."""
    
    def __init__(self, query: Dict[str, Any] = None):
        self.query = query or {}
    
    # Comparison operators
    @staticmethod
    def rating_gte(value: int) -> 'Q':
        """Rating >= value."""
        return Q({"rating": {"$gte": value}})
    
    @staticmethod
    def rating_lte(value: int) -> 'Q':
        """Rating <= value."""
        return Q({"rating": {"$lte": value}})
    
    @staticmethod
    def name_contains(pattern: str) -> 'Q':
        """Name contains pattern (case-insensitive)."""
        return Q({"name": {"$regex": pattern, "$options": "i"}})
    
    @staticmethod
    def has_tag(tag_id: ObjectId) -> 'Q':
        """Has specific tag."""
        return Q({"tag_ids": tag_id})
    
    @staticmethod
    def exclude_tag(tag_id: ObjectId) -> 'Q':
        """Excludes specific tag."""
        return Q({"tag_ids": {"$ne": tag_id}})
    
    @staticmethod
    def in_directory(path: str) -> 'Q':
        """File in directory (regex)."""
        return Q({"path": {"$regex": f"^{path}"}})
    
    @staticmethod
    def file_type(ftype: str) -> 'Q':
        """File type equals."""
        return Q({"file_type": ftype})
    
    @staticmethod
    def extension_in(extensions: List[str]) -> 'Q':
        """Extension in list."""
        return Q({"extension": {"$in": extensions}})
    
    # Logical operators
    @staticmethod
    def AND(*queries: 'Q') -> 'Q':
        """AND multiple queries."""
        return Q({"$and": [q.query for q in queries]})
    
    @staticmethod
    def OR(*queries: 'Q') -> 'Q':
        """OR multiple queries."""
        return Q({"$or": [q.query for q in queries]})
    
    @staticmethod
    def NOT(query: 'Q') -> 'Q':
        """NOT query."""
        return Q({"$nor": [query.query]})


class QueryBuilder:
    """
    Fluent query builder for FileRecord.
    
    Supports:
    - Compound queries with AND/OR/NOT
    - Vector similarity search
    - Sorting and pagination
    - Aggregation
    """
    
    def __init__(self):
        self._query = {}
        self._vector_query = None
        self._vector_threshold = None
        self._sort = []
        self._limit = None
        self._skip = 0
    
    def where(self, query: Q) -> 'QueryBuilder':
        """
        Add query condition.
        
        Args:
            query: Q expression
            
        Returns:
            Self for chaining
        """
        if not self._query:
            self._query = query.query
        else:
            # Combine with existing query (AND)
            self._query = {"$and": [self._query, query.query]}
        
        return self
    
    def AND(self, *queries: Q) -> 'QueryBuilder':
        """
        Add AND condition.
        
        Args:
            queries: Q expressions
            
        Returns:
            Self for chaining
        """
        return self.where(Q.AND(*queries))
    
    def OR(self, *queries: Q) -> 'QueryBuilder':
        """
        Add OR condition.
        
        Args:
            queries: Q expressions
            
        Returns:
            Self for chaining
        """
        return self.where(Q.OR(*queries))
    
    def NOT(self, query: Q) -> 'QueryBuilder':
        """
        Add NOT condition.
        
        Args:
            query: Q expression
            
        Returns:
            Self for chaining
        """
        return self.where(Q.NOT(query))
    
    def vector_similar(
        self,
        vector: List[float],
        threshold: float = 0.8
    ) -> 'QueryBuilder':
        """
        Add vector similarity search.
        
        Args:
            vector: Query vector
            threshold: Similarity threshold
            
        Returns:
            Self for chaining
        """
        self._vector_query = vector
        self._vector_threshold = threshold
        return self
    
    def order_by(self, field: str, descending: bool = False) -> 'QueryBuilder':
        """
        Add sort order.
        
        Args:
            field: Field to sort by
            descending: Sort descending
            
        Returns:
            Self for chaining
        """
        direction = -1 if descending else 1
        self._sort.append((field, direction))
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """
        Limit results.
        
        Args:
            count: Max results
            
        Returns:
            Self for chaining
        """
        self._limit = count
        return self
    
    def skip(self, count: int) -> 'QueryBuilder':
        """
        Skip results (pagination).
        
        Args:
            count: Number to skip
            
        Returns:
            Self for chaining
        """
        self._skip = count
        return self
    
    def get_query(self) -> Dict[str, Any]:
        """Get built query."""
        return self._query
    
    async def execute(self, vector_service=None):
        """
        Execute query.
        
        Args:
            vector_service: VectorService for hybrid search
            
        Returns:
            List of FileRecord
        """
        from src.ucorefs.models.file_record import FileRecord
        
        # Hybrid search if vector query specified
        if self._vector_query and vector_service:
            # Get candidates from vector search
            vector_results = await vector_service.search(
                "file_embeddings",
                self._vector_query,
                filters=self._build_vector_filters(),
                limit=self._limit or 100
            )
            
            # Filter by threshold
            file_ids = [
                r["file_id"] for r in vector_results 
                if r["score"] >= self._vector_threshold
            ]
            
            # Add to query
            if file_ids:
                id_query = {"_id": {"$in": file_ids}}
                if self._query:
                    self._query = {"$and": [self._query, id_query]}
                else:
                    self._query = id_query
        
        # Execute MongoDB query
        results = await FileRecord.find(
            self._query,
            sort=self._sort if self._sort else None,
            limit=self._limit,
            skip=self._skip
        )
        
        return results
    
    def _build_vector_filters(self) -> Optional[Dict[str, Any]]:
        """Build ChromaDB filters from query."""
        # Simplified - would need full query translation
        return None
