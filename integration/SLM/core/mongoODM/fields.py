from bson import ObjectId

class Field:
    """Base class for all ODM fields."""

    def __init__(self, required=False, default=None, db_field=None, unique=False, index=False):
        self.required = required
        self.default = default
        self.db_field = db_field  # The name of the field in the database
        self.unique = unique
        self.index = index
        self.owner_document = None
        self.name = None # will be set by the metaclass

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._data.get(self.name)

    def __set__(self, instance, value):
        if instance is None:
            return
        instance._data[self.name] = value

    def to_mongo(self, value):
        """Convert a Python value to a MongoDB-safe type."""
        return value

    def from_mongo(self, value):
        """Convert a value from MongoDB to a Python type."""
        return value

    def validate(self, value):
        """Validate the value for this field."""
        if self.required and value is None:
            if self.default is None:
                raise ValueError(f"Field '{self.name}' is required.")

class StringField(Field):
    """A field that stores a string."""

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"Value for '{self.name}' must be a string.")

class IntField(Field):
    """A field that stores an integer."""

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, int):
            raise TypeError(f"Value for '{self.name}' must be an integer.")

class FloatField(Field):
    """A field that stores a float."""

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError(f"Value for '{self.name}' must be a float or an integer.")

class BooleanField(Field):
    """A field that stores a boolean."""

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"Value for '{self.name}' must be a boolean.")

class DateTimeField(Field):
    """A field that stores a datetime object."""
    from datetime import datetime

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, self.datetime):
            raise TypeError(f"Value for '{self.name}' must be a datetime object.")

class ListField(Field):
    """A field that stores a list of a specific type."""

    def __init__(self, field_type, **kwargs):
        if not isinstance(field_type, Field):
            raise TypeError("field_type must be an instance of Field")
        self.field_type = field_type
        super().__init__(**kwargs)

    def validate(self, value):
        super().validate(value)
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Value for '{self.name}' must be a list.")
            for item in value:
                self.field_type.validate(item)

    def to_mongo(self, value):
        if value is None:
            return None
        return [self.field_type.to_mongo(item) for item in value]

    def from_mongo(self, value):
        if value is None:
            return None
        return [self.field_type.from_mongo(item) for item in value]

class DictField(Field):
    """A field that stores a dictionary."""

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"Value for '{self.name}' must be a dictionary.")

class EmbeddedDocumentField(Field):
    """A field that stores a nested document."""

    def __init__(self, document_type, **kwargs):
        # Late import to avoid circular dependency
        from .documents import EmbeddedDocument
        if not issubclass(document_type, EmbeddedDocument):
            raise TypeError("document_type must be a subclass of EmbeddedDocument")
        self.document_type = document_type
        super().__init__(**kwargs)

    def validate(self, value):
        super().validate(value)
        if value is not None and not isinstance(value, self.document_type):
            raise TypeError(f"Value for '{self.name}' must be an instance of {self.document_type.__name__}.")

    def to_mongo(self, value):
        if value is None:
            return None
        return value.to_mongo()

    def from_mongo(self, value):
        if value is None:
            return None
        return self.document_type.from_mongo(value)

class ReferenceField(Field):
    """
    A field that stores a reference to another document.
    Supports lazy resolution for self-referencing models.
    """
    def __init__(self, document_type, **kwargs):
        super().__init__(**kwargs)
        # Store the type, but don't validate it yet to avoid circular import issues
        self.document_type_name = document_type if isinstance(document_type, str) else document_type.__name__
        self._document_type = None if isinstance(document_type, str) else document_type

    @property
    def document_type(self):
        """Lazily resolve the document type to avoid circular dependencies."""
        if self._document_type is None:
            from . import DOCUMENT_REGISTRY
            for doc_class in DOCUMENT_REGISTRY:
                if doc_class.__name__ == self.document_type_name:
                    self._document_type = doc_class
                    break
            if self._document_type is None:
                raise TypeError(f"Could not resolve document type '{self.document_type_name}'.")
        return self._document_type

    def __set__(self, instance, value):
        if instance is None:
            return
        # Allow setting the document instance or an ObjectId
        if value is not None and not isinstance(value, (self.document_type, ObjectId)):
             raise TypeError(f"Value for '{self.name}' must be an instance of {self.document_type.__name__} or an ObjectId.")
        instance._data[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        value = instance._data.get(self.name)
        if isinstance(value, self.document_type):
            return value
        
        # If it's an ID, fetch the document
        if value is not None:
            assert self.document_type.objects is not None
            doc = self.document_type.objects.find_one({'_id': value})
            instance._data[self.name] = doc # Cache the fetched document
            return doc
        return None

    def to_mongo(self, value):
        if value is None:
            return None
        if isinstance(value, self.document_type):
            if not hasattr(value, 'pk') or value.pk is None:
                raise ValueError("Cannot create a reference to an unsaved document.")
            return value.pk
        return value # Assume it's already an ObjectId

    def from_mongo(self, value):
        # Return the ObjectId, __get__ will handle lazy loading
        return value

class ReverseReferenceField(Field):
    """
    A field for creating a one-to-many relationship.
    This field does not store any data in the document itself. Instead, it provides
    a query to fetch all documents from another collection that reference this one.
    """
    def __init__(self, document_type_name, field_name, **kwargs):
        self.document_type_name = document_type_name
        self.document_type = None  # Will be resolved later
        self.field_name = field_name
        # This field is virtual and should not be saved
        kwargs['db_field'] = None 
        super().__init__(**kwargs)

    def _resolve_document_type(self):
        """Finds the document class from the registry based on its name."""
        if self.document_type is None:
            from . import DOCUMENT_REGISTRY
            for doc_class in DOCUMENT_REGISTRY:
                if doc_class.__name__ == self.document_type_name:
                    self.document_type = doc_class
                    return
            raise TypeError(f"Could not resolve document type '{self.document_type_name}' for ReverseReferenceField.")

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        self._resolve_document_type()
        
        assert self.document_type is not None
        manager = self.document_type.objects
        if manager is None:
            raise TypeError(f"QuerySet manager not available for {self.document_type.__name__}.")

        if instance.pk is None:
            # Cannot query for references if the document is not saved
            return manager.find({self.field_name: None})

        # Return a QuerySet that finds all related documents
        return manager.find({self.field_name: instance.pk})

    def __set__(self, instance, value):
        # This field is read-only
        raise AttributeError("Cannot set a ReverseReferenceField.")

    def to_mongo(self, value):
        # This field is not stored in the database
        return None

    def from_mongo(self, value):
        # This field is not loaded from the database
        return None

class GenericReferenceField(Field):
    """A field that stores a reference to a document in any collection."""

    def __set__(self, instance, value):
        if instance is None:
            return

        # Late import to avoid circular dependency
        from .documents import Document
        
        # Allow setting from a raw dict during deserialization or a Document instance
        if value is not None and not isinstance(value, (Document, dict)):
            raise TypeError(f"Value for '{self.name}' must be an instance of Document or a reference dict.")
        
        if isinstance(value, dict):
            if not ('pk' in value and '_cls' in value):
                raise TypeError(f"Invalid reference dict for '{self.name}'. Must contain 'pk' and '_cls'.")

        instance._data[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        value = instance._data.get(self.name)
        
        # If the document object is already cached, return it
        from .documents import Document
        if isinstance(value, Document):
            return value
        
        # If it's a reference dict {'pk': ..., '_cls': ...}, fetch the document
        if isinstance(value, dict) and 'pk' in value and '_cls' in value:
            from . import DOCUMENT_REGISTRY
            doc_cls_name = value['_cls']
            doc_class = next((cls for cls in DOCUMENT_REGISTRY if cls.__name__ == doc_cls_name), None)

            if doc_class and hasattr(doc_class, 'objects') and doc_class.objects:
                doc = doc_class.objects.find_one({'_id': value['pk']})
                instance._data[self.name] = doc  # Cache the fetched document
                return doc
        return None

    def to_mongo(self, value):
        if value is None:
            return None
        
        from .documents import Document
        if isinstance(value, Document):
            if not hasattr(value, 'pk') or value.pk is None:
                raise ValueError("Cannot create a generic reference to an unsaved document.")
            return {'pk': value.pk, '_cls': value.__class__.__name__}
        
        # If the value is already a dict, assume it's a valid reference
        if isinstance(value, dict) and 'pk' in value and '_cls' in value:
            return value
            
        raise TypeError(f"Value for '{self.name}' must be a Document instance or a valid reference dict.")

    def from_mongo(self, value):
        # Return the reference dict, __get__ will handle lazy loading
        return value
