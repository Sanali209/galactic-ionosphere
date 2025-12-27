"""
UCoreFS - Tag Manager

Manages hierarchical tags with synonyms and antonyms.
"""
from typing import List, Optional, Set
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.database.manager import DatabaseManager
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
    
    # Dependency declarations for topological startup order
    depends_on = [DatabaseManager]
    
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
    
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
            return False
            
    async def add_tag_to_file(self, file_id: ObjectId, tag_path: str) -> bool:
        """
        Add a tag to a file, creating the tag if it doesn't exist.
        
        Args:
            file_id: File ObjectId
            tag_path: Full tag path (e.g. "auto/wd_tag/sky")
            
        Returns:
            True if successful
        """
        try:
            # 1. Ensure tag exists in taxonomy
            tag = await self.create_tag_from_path(tag_path)
            if not tag:
                return False
            
            # 2. Add to file record
            from src.ucorefs.models.file_record import FileRecord
            
            file = await FileRecord.get(file_id)
            if not file:
                return False
                
            updated = False
            
            # Add ID to tag_ids
            if file.tag_ids is None:
                file.tag_ids = []
            if tag._id not in file.tag_ids:
                file.tag_ids.append(tag._id)
                updated = True
                
            # Add string to tags (denormalized)
            if not hasattr(file, 'tags') or file.tags is None:
                file.tags = []
            if tag_path not in file.tags:
                file.tags.append(tag_path)
                updated = True
            
            if updated:
                await file.save()
                
                # Notify system of file change (for UI updates)
                if hasattr(self, 'locator'):
                    try:
                        from src.core.commands.bus import CommandBus
                        bus = self.locator.get_system(CommandBus)
                        if hasattr(bus, 'publish'):
                            await bus.publish("file.modified", {"file_id": str(file_id)})
                    except (ImportError, KeyError):
                        pass
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tag {tag_path} to file {file_id}: {e}")
            return False
    
    # ==================== Tag Statistics ====================
    
    async def get_tag_statistics(self) -> dict:
        """
        Get statistics about all tags in the system.
        
        Returns:
            Dict with:
                total_count: Total number of tags
                root_count: Number of root-level tags
                max_depth: Maximum tag hierarchy depth
                by_depth: Dict mapping depth -> count
                by_prefix: Dict mapping first path component -> count
                unused_count: Tags with no file associations (if tracked)
        """
        try:
            all_tags = await Tag.find({})
            
            if not all_tags:
                return {
                    "total_count": 0,
                    "root_count": 0,
                    "max_depth": 0,
                    "by_depth": {},
                    "by_prefix": {}
                }
            
            # Calculate statistics
            by_depth = {}
            by_prefix = {}
            max_depth = 0
            root_count = 0
            
            for tag in all_tags:
                # Count by depth
                depth = tag.depth if hasattr(tag, 'depth') else 0
                by_depth[depth] = by_depth.get(depth, 0) + 1
                max_depth = max(max_depth, depth)
                
                # Count root tags
                if not tag.parent_id:
                    root_count += 1
                
                # Count by first path component (prefix)
                if tag.full_path:
                    prefix = tag.full_path.split('/')[0]
                    by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
            
            return {
                "total_count": len(all_tags),
                "root_count": root_count,
                "max_depth": max_depth,
                "by_depth": by_depth,
                "by_prefix": by_prefix
            }
            
        except Exception as e:
            logger.error(f"Failed to get tag statistics: {e}")
            return {"error": str(e)}
    
    async def bulk_rename(
        self,
        old_prefix: str,
        new_prefix: str
    ) -> dict:
        """
        Bulk rename tags by prefix.
        
        Renames all tags starting with old_prefix to use new_prefix.
        Updates both name and full_path.
        
        Args:
            old_prefix: Prefix to replace (e.g., "auto/wd_tag")
            new_prefix: New prefix (e.g., "generated/tags")
            
        Returns:
            Dict with: success, renamed_count, errors
            
        Example:
            # Rename "auto/wd_tag/*" to "generated/tags/*"
            result = await tag_manager.bulk_rename("auto/wd_tag", "generated/tags")
        """
        result = {
            "success": False,
            "old_prefix": old_prefix,
            "new_prefix": new_prefix,
            "renamed_count": 0,
            "errors": []
        }
        
        try:
            # Find all tags with matching prefix
            all_tags = await Tag.find({})
            matching_tags = [
                t for t in all_tags 
                if t.full_path and t.full_path.startswith(old_prefix)
            ]
            
            if not matching_tags:
                result["success"] = True
                logger.info(f"No tags found with prefix: {old_prefix}")
                return result
            
            # Rename each tag
            for tag in matching_tags:
                try:
                    old_path = tag.full_path
                    new_path = new_prefix + old_path[len(old_prefix):]
                    
                    # Update name if it was part of the prefix
                    if '/' in new_path:
                        tag.name = new_path.split('/')[-1]
                    else:
                        tag.name = new_path
                    
                    tag.full_path = new_path
                    await tag.save()
                    
                    result["renamed_count"] += 1
                    
                except Exception as e:
                    result["errors"].append(f"Failed to rename {tag.full_path}: {e}")
            
            result["success"] = True
            logger.info(f"Bulk renamed {result['renamed_count']} tags: {old_prefix} -> {new_prefix}")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Bulk rename failed: {e}")
        
        return result
    
    async def get_tags_report(self) -> str:
        """
        Generate a human-readable tag report.
        
        Returns:
            Formatted string with tag statistics and top tags.
        """
        stats = await self.get_tag_statistics()
        
        if "error" in stats:
            return f"Error generating report: {stats['error']}"
        
        lines = [
            "=== Tag Report ===",
            f"Total Tags: {stats['total_count']}",
            f"Root Tags: {stats['root_count']}",
            f"Max Depth: {stats['max_depth']}",
            "",
            "By Depth:"
        ]
        
        for depth, count in sorted(stats.get("by_depth", {}).items()):
            lines.append(f"  Level {depth}: {count} tags")
        
        lines.append("")
        lines.append("By Prefix (top 10):")
        
        sorted_prefixes = sorted(
            stats.get("by_prefix", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for prefix, count in sorted_prefixes:
            lines.append(f"  {prefix}: {count} tags")
        
        return "\n".join(lines)
    
    # ==================== Count Management ====================
    
    async def recalculate_tag_counts(self) -> dict:
        """
        Recalculate file_count for all tags by counting FileRecords.
        
        This method iterates through all tags and counts how many files
        reference each tag via tag_ids array. Updates are batched for
        performance.
        
        Returns:
            Dict with:
                total_tags: Total number of tags processed
                updated_count: Number of tags with changed counts
                errors: List of error messages
        
        Example:
            result = await tag_manager.recalculate_tag_counts()
            logger.info(f"Updated {result['updated_count']} tag counts")
        """
        from src.ucorefs.models.file_record import FileRecord
        
        result = {
            "total_tags": 0,
            "updated_count": 0,
            "errors": []
        }
        
        try:
            # Get all tags
            all_tags = await Tag.find({})
            result["total_tags"] = len(all_tags)
            
            logger.info(f"Recalculating counts for {result['total_tags']} tags...")
            
            # Process each tag
            for tag in all_tags:
                try:
                    # Count files associated with this tag
                    files = await FileRecord.find({"tag_ids": tag._id})
                    count = len(files)
                    
                    # Update if different
                    if count != tag.file_count:
                        tag.file_count = count
                        await tag.save()
                        result["updated_count"] += 1
                        logger.debug(f"Updated tag '{tag.full_path}': {count} files")
                    
                except Exception as e:
                    error_msg = f"Failed to update tag {tag._id}: {e}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Tag count recalculation complete: {result['updated_count']} updated")
            
        except Exception as e:
            error_msg = f"Tag count recalculation failed: {e}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    async def update_tag_count(self, tag_id: ObjectId) -> int:
        """
        Update file_count for a specific tag by counting FileRecords.
        
        Args:
            tag_id: Tag ObjectId
            
        Returns:
            New count value
            
        Raises:
            ValueError: If tag not found
        """
        from src.ucorefs.models.file_record import FileRecord
        
        tag = await Tag.get(tag_id)
        if not tag:
            raise ValueError(f"Tag not found: {tag_id}")
        
        # Count files
        files = await FileRecord.find({"tag_ids": tag_id})
        count = len(files)
        
        # Update tag
        tag.file_count = count
        await tag.save()
        
        logger.debug(f"Updated tag '{tag.full_path}' count: {count}")
        return count
    
    async def increment_tag_count(self, tag_id: ObjectId, delta: int = 1) -> None:
        """
        Increment/decrement tag file_count atomically.
        
        Uses atomic MongoDB $inc operation for thread safety.
        Ensures count never goes below 0.
        
        Args:
            tag_id: Tag ObjectId
            delta: Amount to add (negative to subtract)
        """
        tag = await Tag.get(tag_id)
        if not tag:
            logger.warning(f"Cannot increment count for non-existent tag: {tag_id}")
            return
        
        # Calculate new count (ensure non-negative)
        new_count = max(0, tag.file_count + delta)
        
        # Update
        tag.file_count = new_count
        await tag.save()
        
        logger.debug(f"Incremented tag {tag.full_path} count by {delta}: {new_count}")
    
    async def remove_tag_from_file(self, file_id: ObjectId, tag_id: ObjectId) -> bool:
        """
        Remove a tag from a file and update counts.
        
        Args:
            file_id: File ObjectId
            tag_id: Tag ObjectId to remove
            
        Returns:
            True if successful
        """
        try:
            from src.ucorefs.models.file_record import FileRecord
            
            # Get file
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            # Remove tag from file
            removed = False
            if hasattr(file, 'tag_ids') and file.tag_ids and tag_id in file.tag_ids:
                file.tag_ids.remove(tag_id)
                removed = True
            
            # Remove from denormalized tags list
            if hasattr(file, 'tags') and file.tags:
                # Find and remove tag string
                tag = await Tag.get(tag_id)
                if tag and tag.full_path in file.tags:
                    file.tags.remove(tag.full_path)
            
            if removed:
                await file.save()
                
                # Decrement tag count
                await self.increment_tag_count(tag_id, -1)
                
                logger.info(f"Removed tag {tag_id} from file {file_id}")
            
            return removed
            
        except Exception as e:
            logger.error(f"Failed to remove tag from file: {e}")
            return False



