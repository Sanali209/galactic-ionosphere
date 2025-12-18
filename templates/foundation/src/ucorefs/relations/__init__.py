"""UCoreFS Relations Package."""
from src.ucorefs.relations.models import Relation, RelationType
from src.ucorefs.relations.service import RelationService

__all__ = [
    "Relation",
    "RelationType",
    "RelationService",
]
