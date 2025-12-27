"""
UCoreFS Vectors Package.

Vector storage and similarity search using FAISS + MongoDB.
"""
from src.ucorefs.vectors.service import VectorService
from src.ucorefs.vectors.faiss_service import FAISSIndexService
from src.ucorefs.vectors.models import EmbeddingRecord

__all__ = [
    "VectorService",
    "FAISSIndexService",
    "EmbeddingRecord",
]

