"""
UCoreFS - Relation Service

Service for managing file relations.
"""
from typing import List, Optional, Dict, Any
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.relations.models import Relation, RelationType


class RelationService(BaseSystem):
    """
    File relationship tracking service.
    
    Manages relationships between files (duplicates, parents, variations, etc).
    """
    
    depends_on = ["DatabaseManager", "ThumbnailService"]
    
    async def initialize(self) -> None:
        """Initialize relation service."""
        logger.info("RelationService initializing")
        
        # Initialize default relation types
        await self._init_default_types()
        
        await super().initialize()
        logger.info("RelationService ready")
    
    async def shutdown(self) -> None:
        """Shutdown relation service."""
        logger.info("RelationService shutting down")
        await super().shutdown()
    
    async def _init_default_types(self) -> None:
        """Initialize default relation types."""
        default_types = [
            {
                "type_name": "image-image",
                "description": "Image to image relation",
                "subtypes": ["duplicate", "near_duplicate", "variant", "wrong"]
            },
            {
                "type_name": "image-detection",
                "description": "Image to detection relation",
                "subtypes": ["contains", "wrong"]
            }
        ]
        
        for type_def in default_types:
            existing = await RelationType.find_one({"type_name": type_def["type_name"]})
            if not existing:
                rel_type = RelationType(**type_def)
                await rel_type.save()
                logger.debug(f"Created relation type: {type_def['type_name']}")
    
    async def create_relation(
        self,
        source_id: ObjectId,
        target_id: ObjectId,
        relation_type: str,
        subtype: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Relation:
        """
        Create a new relation.
        
        Args:
            source_id: Source file ID
            target_id: Target file ID
            relation_type: Type of relation
            subtype: Relation subtype
            payload: Additional data
            
        Returns:
            Created Relation
        """
        # Check if relation already exists
        existing = await Relation.find_one({
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "subtype": subtype
        })
        
        if existing:
            logger.debug(f"Relation already exists: {source_id} -> {target_id}")
            return existing
        
        # Get relation type ID
        rel_type = await RelationType.find_one({"type_name": relation_type})
        
        # Create relation
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type_id=rel_type._id if rel_type else None,
            relation_type=relation_type,
            subtype=subtype,
            payload=payload or {}
        )
        
        await relation.save()
        logger.info(f"Created relation: {relation_type}/{subtype}")
        
        return relation
    
    async def mark_wrong(self, relation_id: ObjectId) -> bool:
        """
        Mark a relation as wrong.
        
        Args:
            relation_id: Relation ObjectId
            
        Returns:
            True if successful
        """
        try:
            relation = await Relation.get(relation_id)
            if relation:
                relation.is_valid = False
                relation.subtype = "wrong"
                await relation.save()
                logger.info(f"Marked relation {relation_id} as wrong")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to mark relation wrong: {e}")
            return False
    
    async def find_relations(
        self,
        file_id: ObjectId,
        relation_type: Optional[str] = None,
        exclude_wrong: bool = True
    ) -> List[Relation]:
        """
        Find all relations for a file.
        
        Args:
            file_id: File ObjectId
            relation_type: Filter by type (optional)
            exclude_wrong: Exclude wrong relations
            
        Returns:
            List of Relation records
        """
        query = {
            "$or": [
                {"source_id": file_id},
                {"target_id": file_id}
            ]
        }
        
        if relation_type:
            query["relation_type"] = relation_type
        
        if exclude_wrong:
            query["is_valid"] = True
        
        return await Relation.find(query)
    
    async def add_relation_subtype(
        self,
        type_name: str,
        subtype: str
    ) -> bool:
        """
        Add a new subtype to a relation type.
        
        Args:
            type_name: Relation type name
            subtype: New subtype to add
            
        Returns:
            True if successful
        """
        try:
            rel_type = await RelationType.find_one({"type_name": type_name})
            if not rel_type:
                return False
            
            if subtype not in rel_type.subtypes:
                rel_type.subtypes.append(subtype)
                await rel_type.save()
                logger.info(f"Added subtype '{subtype}' to {type_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add subtype: {e}")
            return False
