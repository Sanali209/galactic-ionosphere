from src.core.database.models.base import BaseEntity
from src.core.database.orm import FieldPropInfo

class ImageRecord(BaseEntity):
    """
    Represents a physical image file on disk.
    """
    # File Info
    path = FieldPropInfo("path", default="", field_type=str) # Folder path
    filename = FieldPropInfo("filename", default="", field_type=str)
    ext = FieldPropInfo("ext", default="", field_type=str)
    
    # Content Identity
    content_md5 = FieldPropInfo("content_md5", default="", field_type=str)
    size_bytes = FieldPropInfo("size_bytes", default=0, field_type=int)
    
    # Dimensions (stored as dict {w, h})
    dimensions = FieldPropInfo("dimensions", default={}, field_type=dict)
    
    # Metadata (XMP/Exif raw data)
    xmp_data = FieldPropInfo("xmp_data", default={}, field_type=dict)

    @property
    def full_path(self) -> str:
        # Assuming path ends with / or we join properly
        import os
        return os.path.join(self.path, self.filename)
