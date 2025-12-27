"""
UCoreFS Search Package.

Unified search combining MongoDB and FAISS vector similarity.
"""
from src.ucorefs.search.service import SearchService, SearchQuery, SearchResult

__all__ = [
    "SearchService",
    "SearchQuery",
    "SearchResult",
]
