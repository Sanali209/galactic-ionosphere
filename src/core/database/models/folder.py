from src.core.database.orm import CollectionRecord, FieldPropInfo

class FolderRecord(CollectionRecord, table="folders"):
    """
    Represents a folder in the filesystem, synced to DB.
    """
    path = FieldPropInfo("path", default="", field_type=str)
    name = FieldPropInfo("name", default="", field_type=str)
    parent_path = FieldPropInfo("parent_path", default=None, field_type=str) # or None
    is_expanded = FieldPropInfo("is_expanded", default=False, field_type=bool)

    # Indexes for fast lookup
    _indexes = ["path", "parent_path"]

