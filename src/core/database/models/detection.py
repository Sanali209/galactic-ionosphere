from typing import List, Optional
from src.core.database.models.base import BaseEntity
from src.core.database.orm import FieldPropInfo
from bson import ObjectId

class Detection(BaseEntity):
    """
    Represents a specific crop/region on an image found by an OD model.
    """
    parent_image_id = FieldPropInfo("parent_image_id", field_type=ObjectId)
    box = FieldPropInfo("box", field_type=list) # [x, y, w, h]
    class_label = FieldPropInfo("class_label", default="unknown", field_type=str)
    confidence = FieldPropInfo("confidence", default=0.0, field_type=float)
    
    object_id = FieldPropInfo("object_id", default=None, field_type=ObjectId)

    # In our custom ORM, "type_discriminator" usually handles polymorphism if in same collection.
    # If we want separate collection, we might need to override.
    # BaseEntity hardcodes table="gallery_entities". So they share collection.
    
class DetectedObject(BaseEntity):
    """
    Represents a distinct entity (e.g., 'Person A', 'My Cat').
    """
    name = FieldPropInfo("name", default="", field_type=str)
    type = FieldPropInfo("type", default="generic", field_type=str)

