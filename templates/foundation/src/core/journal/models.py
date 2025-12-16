from datetime import datetime
from ..database.orm import CollectionRecord, StringField, DictField

class JournalEntry(CollectionRecord):
    """
    Represents a log entry in the system (User Action, System Event, Error).
    """
    _collection_name = "journal"
    
    # We store timestamp as ISO string or int? 
    # ORM doesn't have DateTimeField yet. Let's use String or Int.
    # original had default=None (probably relied on python object).
    # Since lightweight ORM stores what you give it, let's use StringField for ISO format.
    timestamp = StringField() 
    
    level = StringField(default="INFO")
    source = StringField(default="System")
    message = StringField(default="")
    details = DictField()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
