"""
UCoreFS - Tag Manager

Manages hierarchical tags with synonyms and antonyms.
"""
from typing import List, Optional, Set
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.tags.models import Tag


class TagManager(BaseSystem):
    """
    Tag management service.
    
    Features:
    - Hierarchical tag creation
    - Synonym/antonym relationships
    - Tag search with synonym expansion
    - Batch tagging
    """
    
    async def initialize(self) -> None:
        """Initialize tag manager."""
        logger.info("TagManager initializing")
        await super().initialize()
        logger.info("TagManager ready")
    
    async def shutdown(self) -> None:
        """Shutdown tag manager."""
        logger.info("TagManager shutting down")
        await super().shutdown()
    
    async def create_tag(
        self,
        name: str,
        parent_id: Optional[ObjectId] = None,
        color: str = ""
    ) -> Tag:
        """
        Create a new tag.
        
        Args:
            name: Tag name
            parent_id: Parent tag ID (for hierarchy)
            color: Tag color for UI
            
        Returns:
            Created Tag
        """
        # Build full path
        full_path = name
        depth = 0
        
        if parent_id:
            parent = await Tag.get(parent_id)
            if parent:
                full_path = f"{parent.full_path}/{name}"
                depth = parent.depth + 1
        
        # Check if tag exists
        existing = await Tag.find_one({"full_path": full_path})
        if existing:
            logger.debug(f"Tag already exists: {full_path}")
            return existing
        
        # Create tag
        tag = Tag(
            name=name,
            full_path=full_path,
            parent_id=parent_id,
            depth=depth,
            color=color
        )
        
        # Set MPTT values (simplified - would need proper MPTT algorithm)
        tag.lft = 0
        tag.rgt = 1
        
        await tag.save()
        logger.info(f"Created tag: {full_path}")
        
        return tag
    
    async def add_synonym(
        self,
        tag_id: ObjectId,
        synonym_id: ObjectId
    ) -> bool:
        """
        Add synonym relationship (bidirectional).
        
        Args:
            tag_id: Tag ObjectId
            synonym_id: Synonym tag ObjectId
            
        Returns:
            True if successful
        """
        try:
            tag = await Tag.get(tag_id)
            synonym = await Tag.get(synonym_id)
            
            if not tag or not synonym:
                return False
            
            # Add bidirectional relationship
            if synonym_id not in tag.synonym_ids:
                tag.synonym_ids.append(synonym_id)
                await tag.save()
            
            if tag_id not in synonym.synonym_ids:
                synonym.synonym_ids.append(tag_id)
                await synonym.save()
            
            logger.info(f"Added synonym: {tag.name} <-> {synonym.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add synonym: {e}")
            return False
    
    async def add_antonym(
        self,
        tag_id: ObjectId,
        antonym_id: ObjectId
    ) -> bool:
        """
        Add antonym relationship (bidirectional).
        
        Args:
            tag_id: Tag ObjectId
            antonym_id: Antonym tag ObjectId
            
        Returns:
            True if successful
        """
        try:
            tag = await Tag.get(tag_id)
            antonym = await Tag.get(antonym_id)
            
            if not tag or not antonym:
                return False
            
            # Add bidirectional relationship
            if antonym_id not in tag.antonym_ids:
                tag.antonym_ids.append(antonym_id)
                await tag.save()
            
            if tag_id not in antonym.antonym_ids:
                antonym.antonym_ids.append(tag_id)
                await antonym.save()
            
            logger.info(f"Added antonym: {tag.name} <-> {antonym.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add antonym: {e}")
            return False
    
    async def get_synonyms(self, tag_id: ObjectId) -> List[Tag]:
        """
        Get all synonym tags.
        
        Args:
            tag_id: Tag ObjectId
            
        Returns:
            List of synonym Tags
        """
        tag = await Tag.get(tag_id)
        if not tag or not tag.synonym_ids:
            return []
        
        return await Tag.find({"_id": {"$in": tag.synonym_ids}})
    
    async def expand_search_with_synonyms(
        self,
        tag_ids: List[ObjectId]
    ) -> Set[ObjectId]:
        """
        Expand tag search to include synonyms.
        
        Args:
            tag_ids: List of tag IDs to search
            
        Returns:
            Set of expanded tag IDs (original + synonyms)
        """
        expanded = set(tag_ids)
        
        for tag_id in tag_ids:
            synonyms = await self.get_synonyms(tag_id)
            expanded.update([s._id for s in synonyms])
        
        return expanded
    
    async def check_antonym_conflict(
        self,
        tag_ids: List[ObjectId]
    ) -> List[tuple]:
        """
        Check for antonym conflicts in tag list.
        
        Args:
            tag_ids: List of tag IDs
            
        Returns:
            List of (tag1_id, tag2_id) antonym pairs
        """
        conflicts = []
        
        for i, tag_id in enumerate(tag_ids):
            tag = await Tag.get(tag_id)
            if not tag:
                continue
            
            # Check if any other tag in list is an antonym
            for other_id in tag_ids[i+1:]:
                if other_id in tag.antonym_ids:
                    conflicts.append((tag_id, other_id))
        
        return conflicts
    
    async def get_children(self, tag_id: Optional[ObjectId] = None) -> List[Tag]:
        """
        Get child tags.
        
        Args:
            tag_id: Parent tag ID (None for root tags)
            
        Returns:
            List of child Tags
        """
        return await Tag.find({"parent_id": tag_id})
    
    async def create_tag_from_path(
        self,
        tag_path: str,
        delimiter: str = None,
        color: str = ""
    ) -> Tag:
        """
        Create tag from path string, parsing hierarchy.
        
        Parses delimiters (/, |, \\) to create parent/child tags.
        Example: "Animals/Mammals/Cat" creates:
          - Animals (root)
            - Mammals (child of Animals)
              - Cat (child of Mammals)
        
        Args:
            tag_path: Tag path with delimiters (e.g., "test/sub" or "test|sub")
            delimiter: Optional specific delimiter (auto-detect if None)
            color: Tag color
            
        Returns:
            The leaf (deepest) Tag
        """
        # Auto-detect delimiter
        if delimiter is None:
            for d in ['/', '|', '\\']:
                if d in tag_path:
                    delimiter = d
                    break
        
        # No delimiter found - create simple tag
        if delimiter is None or delimiter not in tag_path:
            return await self.create_tag(name=tag_path.strip(), color=color)
        
        # Split path
        parts = [p.strip() for p in tag_path.split(delimiter) if p.strip()]
        
        if not parts:
            logger.warning(f"Empty tag path after split: {tag_path}")
            return None
        
        # Create each level
        parent_id = None
        current_tag = None
        
        for part in parts:
            current_tag = await self.create_tag(
                name=part,
                parent_id=parent_id,
                color=color if part == parts[-1] else ""  # Color only on leaf
            )
            parent_id = current_tag._id
        
        return current_tag
    
    async def get_all_flat(self) -> List[Tag]:
        """
        Get all tags as flat list.
        
        Returns:
            List of all Tags
        """
        return await Tag.find({})
    
    async def delete_tag(self, tag_id: ObjectId, recursive: bool = True) -> bool:
        """
        Delete a tag and optionally its children.
        
        Args:
            tag_id: Tag ID to delete
            recursive: If True, delete children too
            
        Returns:
            True if successful
        """
        try:
            tag = await Tag.get(tag_id)
            if not tag:
                return False
            
            if recursive:
                # Delete children first
                children = await self.get_children(tag_id)
                for child in children:
                    await self.delete_tag(child._id, recursive=True)
            
            await tag.delete()
            logger.info(f"Deleted tag: {tag.full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
            return False
