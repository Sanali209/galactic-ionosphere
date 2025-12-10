from typing import List, Optional
from bson import ObjectId
from src.core.database.orm import CollectionRecord, FieldPropInfo

class Tag(CollectionRecord, table="tags", indexes=["path", "parent_id"]):
    """
    Hierarchical Tag using Materialized Path.
    Path format: "grandparent_id|parent_id"
    """
    name = FieldPropInfo("name", default="New Tag", field_type=str)
    description = FieldPropInfo("desc", default="", field_type=str)
    
    # Hierarchy
    parent_id = FieldPropInfo("parent_id", default=None, field_type=ObjectId)
    path = FieldPropInfo("path", default="", field_type=str)
    
    @property
    def depth(self) -> int:
        return len(self.path.split("|")) if self.path else 0

class TagManager:
    """
    Service for managing Tag hierarchy operations (Create, Move, Delete).
    """
    @staticmethod
    async def create_tag(name: str, parent: Optional[Tag] = None) -> Tag:
        t = Tag(name=name)
        if parent:
            t.parent_id = parent.id
            # Build path: parent's path + parent's ID
            prefix = parent.path + "|" if parent.path else ""
            t.path = f"{prefix}{str(parent.id)}"
        
        await t.save()
        return t

    @staticmethod
    async def move_tag(tag: Tag, new_parent: Optional[Tag]):
        """
        Moves a tag to a new parent and updates paths of ALL descendants.
        """
        old_path_prefix = f"{tag.path}|{str(tag.id)}" if tag.path else str(tag.id)
        if not tag.path and not tag.parent_id:
             # It was a root
             old_path_prefix = str(tag.id)
        
        # Calculate new path for the tag itself
        new_path = ""
        if new_parent:
            parent_prefix = new_parent.path + "|" if new_parent.path else ""
            new_path = f"{parent_prefix}{str(new_parent.id)}"
            tag.parent_id = new_parent.id
        else:
            tag.parent_id = None
        
        tag.path = new_path
        await tag.save()
        
        # Update all children (Recursive update of paths)
        # Old prefix: "Root|A"
        # New prefix: "Root|B"
        # Child was: "Root|A|Child" -> "Root|B|Child"
        
        # We find all tags that start with the OLD prefix + divider
        # divider is "|"
        
        from src.core.database.manager import db_manager
        coll = Tag.get_collection()
        
        # Construct Regex for finding descendants
        # Ancestors path is stored in 'path'.
        # If I am tag A (id=123), my path was "P".
        # My children have path "P|123".
        # Now my path is "N".
        # Children need path "N|123".
        
        # We need to replace the START of the path string for all descendants.
        # The part to replace is `old_path_prefix`.
        # The replacement is `new_path` + "|" + `str(tag.id)`  Wait.
        
        # tag.id remains same.
        # tag.path changed from P to N.
        # Children typically append tag.id to tag.path.
        # So children had: "P|tag.id|..."
        # Now need:        "N|tag.id|..."
        
        # We can enable multi-update with aggregation pipeline or simple iteration.
        # Iteration is safer for correctness, let's do find and update.
        
        # Find all tags where path starts with "old_path_prefix"
        # old_path_prefix was calculated above as `tag.path|tag.id` BEFORE the save?
        # Oops, we modified `tag.path` above. We lost the old path!
        # We actually calculated `old_path_prefix` at the start of method:
        # `old_path_prefix = f"{tag.path}|{str(tag.id)}"` <-- This used the OLD path. correct.
        
        new_prefix = f"{tag.path}|{str(tag.id)}" if tag.path else str(tag.id)
        
        # MongoDB $regex query
        query = {"path": {"$regex":f"^{old_path_prefix}"}}
        
        # find() is async and returns a List, not an cursor
        children = await Tag.find(query)
        for child in children:
            # Replace prefix
            # child.path starts with old_path_prefix
            remaining = child.path[len(old_path_prefix):]
            child.path = new_prefix + remaining
            await child.save()
