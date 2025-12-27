"""
UCoreFS - Query Builder

Fluent API for building complex MongoDB queries.
Enhanced with Field class for type-safe query expressions.
"""
from typing import Any, Dict, List, Optional, Union
from bson import ObjectId


class Q:
    """
    Query expression builder (Condition).
    
    Allows combining conditions with &, |, and ~ operators.
    
    Example:
        query = (Q.field("age").gte(18) & Q.field("status").eq("active")) | Q.field("role").eq("admin")
    """
    
    def __init__(self, query: Dict[str, Any] = None):
        """Initialize query with optional dict."""
        self.query = query or {}
    
    def __and__(self, other: 'Q') -> 'Q':
        """Logical AND with & operator."""
        return Q({"$and": [self.query, other.query]})
    
    def __or__(self, other: 'Q') -> 'Q':
        """Logical OR with | operator."""
        return Q({"$or": [self.query, other.query]})
    
    def __invert__(self) -> 'Q':
        """Logical NOT with ~ operator."""
        return Q({"$not": self.query})
    
    def __repr__(self) -> str:
        return f"Q({self.query})"
    
    # --- Shortcut factory for Field ---
    @staticmethod
    def field(field_name: str) -> 'Field':
        """Create a Field for building conditions."""
        return Field(field_name)
    
    # --- Legacy static methods for backward compatibility ---
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
    
    # --- Logical operators ---
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


class Field:
    """
    Fluent field accessor for building query conditions.
    
    Ported from SLM/mongoext/collectionQueryDSL.py.
    
    Example:
        Q.field("age").gte(18)             # {"age": {"$gte": 18}}
        Q.field("tags").in_(["a", "b"])    # {"tags": {"$in": ["a", "b"]}}
        Q.field("name").regex("^test", "i")  # {"name": {"$regex": "^test", "$options": "i"}}
    """
    
    def __init__(self, field_name: str):
        """Initialize with field name."""
        self.field_name = field_name
    
    def eq(self, value: Any) -> Q:
        """Equal to value."""
        return Q({self.field_name: value})
    
    def ne(self, value: Any) -> Q:
        """Not equal to value."""
        return Q({self.field_name: {"$ne": value}})
    
    def lt(self, value: Any) -> Q:
        """Less than value."""
        return Q({self.field_name: {"$lt": value}})
    
    def lte(self, value: Any) -> Q:
        """Less than or equal to value."""
        return Q({self.field_name: {"$lte": value}})
    
    def gt(self, value: Any) -> Q:
        """Greater than value."""
        return Q({self.field_name: {"$gt": value}})
    
    def gte(self, value: Any) -> Q:
        """Greater than or equal to value."""
        return Q({self.field_name: {"$gte": value}})
    
    def in_(self, values: List[Any]) -> Q:
        """Value in list."""
        return Q({self.field_name: {"$in": values}})
    
    def nin(self, values: List[Any]) -> Q:
        """Value not in list."""
        return Q({self.field_name: {"$nin": values}})
    
    def exists(self, does_exist: bool = True) -> Q:
        """Field exists or not."""
        return Q({self.field_name: {"$exists": does_exist}})
    
    def regex(self, pattern: str, options: str = "") -> Q:
        """
        Regex match.
        
        Args:
            pattern: Regex pattern
            options: Options like "i" for case-insensitive
        """
        if options:
            return Q({
                self.field_name: {
                    "$regex": pattern,
                    "$options": options
                }
            })
        return Q({self.field_name: {"$regex": pattern}})
    
    def type_(self, bson_type: Union[int, str]) -> Q:
        """
        Field is of BSON type.
        
        Args:
            bson_type: BSON type name or number
        """
        return Q({self.field_name: {"$type": bson_type}})
    
    def mod(self, divisor: int, remainder: int) -> Q:
        """Field modulo equals remainder."""
        return Q({self.field_name: {"$mod": [divisor, remainder]}})
    
    # --- Array operators ---
    def all_(self, values: List[Any]) -> Q:
        """Array contains all elements."""
        return Q({self.field_name: {"$all": values}})
    
    def size(self, size_val: int) -> Q:
        """Array has exact size."""
        return Q({self.field_name: {"$size": size_val}})
    
    def elem_match(self, condition: Q) -> Q:
        """Array element matches condition."""
        return Q({self.field_name: {"$elemMatch": condition.query}})


# Shortcut function matching SLM pattern
def F(field_name: str) -> Field:
    """
    Shortcut to create a Field object.
    
    Same as Q.field(field_name).
    
    Example:
        query = F("name").regex("test", "i") & F("age").gte(18)
    """
    return Field(field_name)




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
            vector_service: VectorService for hybrid search (auto-resolved if None)
            
        Returns:
            List of FileRecord
        """
        from src.ucorefs.models.file_record import FileRecord
        
        # Auto-resolve VectorService from ServiceLocator if needed
        if vector_service is None and self._vector_query:
            try:
                from src.core.locator import sl
                from src.ucorefs.vectors.service import VectorService
                vector_service = sl.get_system(VectorService)
            except (KeyError, ImportError):
                pass  # Vector search not available
        
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
        """
        Build ChromaDB filters from query.
        
        Translates supported MongoDB query patterns to ChromaDB where clauses.
        Supported patterns: file_type, extension (equality and $in).
        
        Returns:
            ChromaDB-compatible filter dict or None if no translatable filters
        """
        if not self._query:
            return None
        
        filters = {}
        
        def extract_filters(query: Dict[str, Any]) -> None:
            """Recursively extract translatable filters from query."""
            for key, value in query.items():
                # Handle direct equality: {"file_type": "image"}
                if key in ("file_type", "extension", "driver_type") and isinstance(value, str):
                    filters[key] = value
                
                # Handle $in operator: {"extension": {"$in": [".jpg", ".png"]}}
                elif key in ("file_type", "extension", "driver_type") and isinstance(value, dict):
                    if "$in" in value:
                        filters[key] = {"$in": value["$in"]}
                    elif "$eq" in value:
                        filters[key] = value["$eq"]
                
                # Handle $and: recurse into each condition
                elif key == "$and" and isinstance(value, list):
                    for sub_query in value:
                        extract_filters(sub_query)
        
        extract_filters(self._query)
        
        if not filters:
            return None
        
        # Convert to ChromaDB where format
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            if isinstance(val, dict) and "$in" in val:
                return {"$or": [{key: {"$eq": v}} for v in val["$in"]]}
            return {key: {"$eq": val}}
        
        # Multiple filters: combine with $and
        where_clauses = []
        for key, val in filters.items():
            if isinstance(val, dict) and "$in" in val:
                where_clauses.append({"$or": [{key: {"$eq": v}} for v in val["$in"]]})
            else:
                where_clauses.append({key: {"$eq": val}})
        
        return {"$and": where_clauses}

