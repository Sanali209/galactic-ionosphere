from datetime import datetime
from ..database.orm import CollectionRecord, StringField, IntField, ListField, DictField, BoolField

class TaskRecord(CollectionRecord):
    _collection_name = "tasks"
    
    name = StringField()
    handler_name = StringField()     # Key to lookup function
    task_args = ListField(StringField()) # Simple string args for now
    
    status = StringField(default="pending") # pending, running, completed, failed, quarantined
    priority = IntField(default=1)  # 0=HIGH, 1=NORMAL, 2=LOW
    progress = IntField(default=0)
    
    result = StringField() # JSON serialized result
    error = StringField()
    error_type = StringField()  # NEW: Categorized error (missing_resource, network, unknown)
    
    # Timestamps
    created_at = IntField()
    started_at = IntField()      # NEW: When status became 'running'
    completed_at = IntField()    # NEW: When status became 'completed'/'failed'
    updated_at = IntField()      # NEW: Last status change
    
    # Recovery metadata
    recovery_count = IntField(default=0)  # NEW: How many times recovered from crash
    interrupted = StringField()           # NEW: Reason for interruption
    retryable = BoolField(default=True)   # NEW: Whether task can be retried

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        now = int(datetime.utcnow().timestamp())
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

