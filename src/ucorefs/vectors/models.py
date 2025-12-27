"""
UCoreFS - Embedding Models

MongoDB models for storing vector embeddings.
Replaces ChromaDB with native MongoDB storage + FAISS for similarity search.
"""
from datetime import datetime
from typing import List, Optional
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class EmbeddingRecord(CollectionRecord):
    """
    Stores vector embeddings in MongoDB.
    
    One record per file per provider (e.g., CLIP, BLIP).
    FAISS builds in-memory index from these records.
    
    Indexes:
    - file_id + provider (unique compound)
    - provider (for bulk loading)
    """
    
    # Foreign key to FileRecord
    file_id: ObjectId = Field(index=True)
    
    # Embedding provider info
    provider: str = Field(index=True)  # "clip", "blip", "mobilenet"
    model_version: str = Field(default="")  # "ViT-B/32"
    
    # The vector itself
    vector: List[float] = Field(default_factory=list)
    dimension: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Settings:
        name = "embedding_records"
        indexes = [
            {"keys": [("file_id", 1), ("provider", 1)], "unique": True},
        ]
    
    def __str__(self) -> str:
        return f"Embedding({self.provider}): file={self.file_id}"
