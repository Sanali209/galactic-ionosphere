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
    async def ensure_from_path(path_str: str, separator: str = "|") -> Tag:
        """
        Ensures a hierarchy of tags exists given a path string (e.g., 'Animals|Mammals|Cats').
        Returns the leaf tag.
        """
        parts = [p.strip() for p in path_str.split(separator) if p.strip()]
        if not parts:
            raise ValueError("Empty tag path")

        current_parent = None

        for name in parts:
            # Find existing child of current_parent with this name
            query = {"name": name}
            if current_parent:
                query["parent_id"] = current_parent.id
            else:
                # Root tag
                query["parent_id"] = None

            found = await Tag.find(query)
            if found:
                current_parent = found[0]
            else:
                # Create
                current_parent = await TagManager.create_tag(name, current_parent)

        return current_parent

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
