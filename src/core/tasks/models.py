from datetime import datetime
from ..database.orm import CollectionRecord, StringField, IntField, ListField, DictField

class TaskRecord(CollectionRecord):
    _collection_name = "tasks"
    
    name = StringField()
    handler_name = StringField()     # Key to lookup function
    task_args = ListField(StringField()) # Simple string args for now, or use DictField for complex
    # Note: To support complex args, we might need a serialized blob or robust DictField support
    
    status = StringField(default="pending") # pending, running, completed, failed
    progress = IntField(default=0)
    
    result = StringField() # JSON serialized result?
    error = StringField()
    
    # created_at = StringField() # DateTimeField not implemented in ORM yet, passing as str or int timestamp
    created_at = IntField()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.created_at is None:
            self.created_at = int(datetime.utcnow().timestamp())
