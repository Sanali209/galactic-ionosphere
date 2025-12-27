"""
Search history ORM model.
"""
from datetime import datetime
from typing import List

from src.core.database.orm import CollectionRecord, Field


class SearchHistory(CollectionRecord):
    """
    Stores search query history with metadata.
    Auto collection name: "search_histories"
    """
    # _collection_name auto-generates as "search_histories"
    
    query: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
    result_count: int = Field(default=0)
    max_results: int = Field(default=20)
    
    # Stored search terms for quick lookup
    tags: List[str] = Field(default_factory=list)
    
    def __str__(self):
        return f"Search: '{self.query}' ({self.result_count} results at {self.timestamp})"
