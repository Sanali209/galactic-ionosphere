from typing import List, Optional
from bson import ObjectId
from src.core.database.orm import CollectionRecord, FieldPropInfo

class BaseEntity(CollectionRecord, table="gallery_entities"):
    """
    Polymorphic root for all gallery items (Images, Detections, Groups).
    Stored in 'gallery_entities' collection.
    """
    # Common Metadata
    rating = FieldPropInfo("rating", default=0, field_type=int)
    label = FieldPropInfo("label", default="none", field_type=str)
    description = FieldPropInfo("desc", default="", field_type=str)
    
    # Tagging (List of Tag ObjectIds)
    tag_ids = FieldPropInfo("tag_ids", default=[], field_type=list)
    
    # Relations (List of serialized Reference objects or just IDs if using simple linking)
    # For now, we keep it simple, complex graph edges go to 'references' collection.
    
    def add_tag(self, tag_id: ObjectId):
        """Helper to add a tag securely."""
        if not isinstance(tag_id, ObjectId):
            tag_id = ObjectId(tag_id)
        self.list_append("tag_ids", tag_id)

    def remove_tag(self, tag_id: ObjectId):
        """Helper to remove a tag."""
        if not isinstance(tag_id, ObjectId):
            tag_id = ObjectId(tag_id)
        
        # We need a list_remove helper in ORM or implement logic here
        # Since ORM doesn't have list_remove yet, we fetch, modify, set.
        current = self.tag_ids
        if tag_id in current:
            new_list = [t for t in current if t != tag_id]
            self.tag_ids = new_list # Triggers reactive event
