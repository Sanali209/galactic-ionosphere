from datetime import datetime
from src.core.database.orm import CollectionRecord, FieldPropInfo

class JournalRecord(CollectionRecord, table="journal"):
    timestamp = FieldPropInfo("timestamp", default=datetime.now, field_type=datetime)
    level = FieldPropInfo("level", default="INFO", field_type=str)
    category = FieldPropInfo("category", default="SYSTEM", field_type=str)
    message = FieldPropInfo("message", default="", field_type=str)
    # Using dict for details allows storing flexible JSON data
    details = FieldPropInfo("details", default={}, field_type=dict)
