from .metaclass import ODMBase
from bson import ObjectId


class BaseDocument(metaclass=ODMBase):
    """
    Base class for all documents.
    Handles data conversion and initialization.
    Now supports dynamic field access and optional caching like MongoRecordWrapper.
    """

    def __init__(self, **kwargs):
        # When re-initializing a cached instance, don't wipe the data
        if not hasattr(self, '_data'):
            self._data = {}
        if not hasattr(self, '_queryset'):
            self._queryset = None
        
        # Initialize caching system - can be disabled for memory-sensitive applications
        if not hasattr(self, 'props_cache'):
            self.props_cache = {}
        if not hasattr(self, '_enable_field_caching'):
            self._enable_field_caching = getattr(self.__class__, '_enable_field_caching', True)

        # Initialize with default values only for new instances
        is_new = 'pk' not in self._data and '_id' not in kwargs and 'pk' not in kwargs
        if is_new:
            for name, field in self._fields.items():
                if field.default is not None:
                    self._data[name] = field.default() if callable(field.default) else field.default

        # Populate with provided data
        for key, value in kwargs.items():
            if key in self._fields:
                setattr(self, key, value)
            elif key == '_id' or key == 'pk':
                self._data['pk'] = value
            else:
                # Allow setting extra data that is not defined in fields
                self._data[key] = value

    @property
    def pk(self):
        return self._data.get('pk')

    def get_field_val(self, field_name, default=None, use_cache=True):
        """
        Get field value from database or cache (wrapper-style).
        Similar to MongoRecordWrapper.get_field_val()
        """
        if use_cache and self._enable_field_caching and field_name in self.props_cache:
            return self.props_cache[field_name]
        
        # Load fresh data from database
        self.get_record_data()
        return self.props_cache.get(field_name, default)

    def set_field_val(self, field_name, value):
        """
        Set field value and immediately save to database (wrapper-style).
        Similar to MongoRecordWrapper.set_field_val()
        """
        data = {field_name: value}
        self.set_record_data(data)

    def get_record_data(self):
        """
        Load document data from database and update cache (wrapper-style).
        Similar to MongoRecordWrapper.get_record_data()
        """
        if self.pk is None:
            return {}
        
        assert self.__class__.objects is not None, "QuerySet manager is not initialized."
        collection = self.__class__.objects._collection
        res = collection.find_one({'_id': self.pk})
        
        if res is None:
            return {}
        
        if self._enable_field_caching:
            self.props_cache = res
        return res

    def set_record_data(self, data):
        """
        Update specific fields in database and refresh cache (wrapper-style).
        Similar to MongoRecordWrapper.set_record_data()
        """
        if self.pk is None:
            raise ValueError("Cannot update document without primary key")
        
        assert self.__class__.objects is not None, "QuerySet manager is not initialized."
        collection = self.__class__.objects._collection
        
        query = {'_id': self.pk}
        collection.update_one(query, {'$set': data}, upsert=True)
        
        # Fire edit event using core event bus (wrapper-style)
        self._fire_edit_event()
        
        # Refresh cache
        self.get_record_data()

    def invalidate_cache(self):
        """Clear the properties cache (wrapper-style)."""
        self.props_cache = {}

    def clear_cache(self):
        """Alias for invalidate_cache for compatibility."""
        self.invalidate_cache()

    # List manipulation methods (wrapper-style)
    def list_get(self, field_name):
        """Get list field value (wrapper-style)."""
        if self._enable_field_caching and field_name in self.props_cache:
            return self.props_cache.get(field_name, [])
        
        data = self.get_record_data()
        return data.get(field_name, [])

    def list_append(self, field_name, value, no_dupes=False):
        """Append value to list field (wrapper-style)."""
        current_list = self.list_get(field_name)
        current_list.append(value)
        
        if no_dupes:
            current_list = list(set(current_list))
        
        self.set_record_data({field_name: current_list})

    def list_extend(self, field_name, values, no_dupes=False):
        """Extend list field with multiple values (wrapper-style)."""
        current_list = self.list_get(field_name)
        current_list.extend(values)
        
        if no_dupes:
            current_list = list(set(current_list))
        
        self.set_record_data({field_name: current_list})

    def list_remove(self, field_name, value):
        """Remove value from list field (wrapper-style)."""
        current_list = set(self.list_get(field_name))
        if value in current_list:
            current_list.remove(value)
            self.set_record_data({field_name: list(current_list)})

    def to_mongo(self):
        """Converts the document to a dictionary for MongoDB."""
        data = {}
        for name, field in self._fields.items():
            value = getattr(self, name)
            if value is not None:
                db_field_name = field.db_field or name
                data[db_field_name] = field.to_mongo(value)

        # Add _cls for polymorphic queries if it's a full Document
        if isinstance(self, Document):
            manager = self.__class__.objects
            if manager and len(manager.polymorphic_map) > 1:
                data['_cls'] = self.__class__.__name__

        return data

    @classmethod
    def from_mongo(cls, data):
        """Creates a document instance from MongoDB data."""
        if data is None:
            return None

        prepared_data = {}
        field_map = {(f.db_field or n): n for n, f in cls._fields.items()}

        for db_key, value in data.items():
            if db_key == '_id':
                prepared_data['pk'] = value
                continue

            field_name = field_map.get(db_key)
            if field_name:
                field = cls._fields[field_name]
                # A better way to check for virtual fields
                if 'ReverseReferenceField' in field.__class__.__name__:
                    continue
                prepared_data[field_name] = field.from_mongo(value)
            else:
                # Keep data that doesn't map to a field
                prepared_data[db_key] = value

        return cls(**prepared_data)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.pk}>"

    def _set_queryset(self, queryset):
        self._queryset = queryset

    def _fire_edit_event(self):
        """Fire edit event using core event bus (wrapper-style)."""
        manager = self._queryset or (self.__class__.objects if hasattr(self.__class__, 'objects') else None)
        if manager and manager._message_bus:
            manager._message_bus.publish(f"document.{self.__class__.__name__.lower()}.on_edit", document=self)


class EmbeddedDocument(BaseDocument):
    """A document that is embedded in another document."""
    pass


class Document(BaseDocument):
    """A document that can be stored in its own collection."""

    # This will be populated by the metaclass
    objects = None

    @classmethod
    def find_one(cls, query):
        """Finds a single document matching the query."""
        assert cls.objects is not None, "QuerySet manager is not initialized."
        return cls.objects.find_one(query)

    @classmethod
    def find(cls, query, sort_query=None):
        """Finds all documents matching the query."""
        assert cls.objects is not None, "QuerySet manager is not initialized."
        return list(cls.objects.find(filter=query, sort=sort_query))

    @classmethod
    def get_by_id(cls, record_id):
        """Gets a document by its primary key."""
        assert cls.objects is not None, "QuerySet manager is not initialized."
        if not isinstance(record_id, ObjectId):
            try:
                record_id = ObjectId(record_id)
            except Exception:
                return None
        return cls.objects.find_one({'_id': record_id})

    @classmethod
    def get_or_create(cls, **kwargs):
        """
        Finds a document with the given kwargs, creating it if it doesn't exist.
        Returns a tuple of (document, created), where created is a boolean.
        """
        assert cls.objects is not None, "QuerySet manager is not initialized."
        
        # Separate query fields from other fields
        query_kwargs = {k: v for k, v in kwargs.items() if k in cls._fields}
        
        instance = cls.objects.find_one(query_kwargs)
        if instance:
            return instance, False
        
        # If not found, create a new one with all kwargs
        instance = cls(**kwargs)
        instance._save_new_document()  # Auto-save new document
        return instance, True

    @classmethod
    def new_record(cls, **kwargs):
        """
        Create new record in database (wrapper-style).
        Similar to MongoRecordWrapper.new_record()
        """
        return cls._create_new(**kwargs)

    @classmethod
    def _create_new(cls, **kwargs):
        """Internal method to create and save a new document."""
        instance = cls(**kwargs)
        instance._save_new_document()  # Auto-save new document
        return instance

    @classmethod
    def create_index(cls, index_name, index_fields):
        """
        Create database index (wrapper-style).
        Similar to MongoRecordWrapper.create_index()
        """
        assert cls.objects is not None, "QuerySet manager is not initialized."
        collection = cls.objects._collection
        indexes = collection.index_information()
        
        if index_name in indexes:
            return
        
        try:
            collection.create_index(index_fields, name=index_name)
        except Exception as e:
            # Use loguru if available, otherwise print
            try:
                from loguru import logger
                logger.error(f"Error creating index {index_name}: {e}")
            except ImportError:
                print(f"Error creating index {index_name}: {e}")

    @classmethod
    def collection(cls):
        """
        Get the MongoDB collection for this document (wrapper-style).
        Similar to MongoRecordWrapper.collection()
        """
        assert cls.objects is not None, "QuerySet manager is not initialized."
        return cls.objects._collection

    def __setattr__(self, name, value):
        """
        Override attribute setting to automatically persist changes to database.
        All field changes are handled in background without requiring explicit save().
        """
        # Set the attribute first
        super().__setattr__(name, value)
        
        # Auto-persist to database if this is a field change on existing document
        if hasattr(self, '_data') and self.pk is not None and name in self._fields:
            try:
                # Use set_field_val for immediate database persistence
                self.set_field_val(name, value)
            except Exception:
                # Silently continue if database operation fails
                pass

    def _save_new_document(self):
        """
        Internal method to save a new document to database.
        Used when document is first created.
        """
        if self.pk is not None:
            return  # Already saved
            
        assert self.objects is not None, "Document class must have objects manager initialized"
        manager = self._queryset or self.objects
        if manager is None:
            return  # Cannot save without manager
        
        data = self.to_mongo()
        result = manager.insert_one(data)
        self._data['pk'] = result.inserted_id
        
        # Fire message bus event
        if manager._message_bus:
            event_name = f"document.{self.__class__.__name__.lower()}.created"
            manager._message_bus.publish(event_name, document=self)
        
        # Ensure the instance is bound to the queryset after saving
        if self._queryset is None:
            self._queryset = manager

    def delete(self):
        """Delete the document from the database."""
        if self.pk is None:
            return

        assert self.objects is not None
        manager = self._queryset or self.objects
        if manager is None:
            raise TypeError(
                "This document cannot be deleted. Is the DatabaseManagerComponent running and the document registered?")

        # Fire delete events using core event bus (wrapper-style)
        if manager._message_bus:
            # Fire specific document delete event
            manager._message_bus.publish(f"document.{self.__class__.__name__.lower()}.on_delete", document=self)
            # Fire global delete event
            manager._message_bus.publish("document.global.on_delete", document=self)
            # Fire existing ODM event for backward compatibility
            manager._message_bus.publish(f"document.{self.__class__.__name__.lower()}.deleted", document_id=self.pk)

        manager.delete_one({'_id': self.pk})
        self._data['pk'] = None

    def delete_rec(self):
        """
        Delete record from database (wrapper-style compatibility).
        Alias for delete() method to match MongoRecordWrapper API.
        """
        self.delete()
