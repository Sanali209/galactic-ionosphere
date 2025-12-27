from datetime import datetime
from SLM.core.mongoODM.documents import Document, EmbeddedDocument
from SLM.core.mongoODM.fields import (
    StringField,
    IntField,
    FloatField,
    DateTimeField,
    ListField,
    DictField,
    BooleanField,
    ReferenceField,
    EmbeddedDocumentField,
    GenericReferenceField
)

class CollectionItem(Document):
    """
    An abstract base class for any item that can be part of a collection,
    such as a file or a tag.
    """
    __abstract__ = True

    rating = IntField(default=0)
    description = StringField()
    tags = ListField(StringField(), default=[])
    is_deleted = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    def __setattr__(self, name, value):
        """
        Override to automatically update updated_at when any field changes.
        Uses new mongoODM automatic persistence - no explicit save() needed.
        """
        # Update timestamp for any field change (except internal fields)
        if name in self._fields and name != 'updated_at':
            super().__setattr__('updated_at', datetime.utcnow())
        
        # Call parent to handle automatic persistence
        super().__setattr__(name, value)

class AIExpertise(EmbeddedDocument):
    """
    An embedded document to store the results from various AI services or "indexers".
    """
    service_name = StringField(required=True)
    backend_name = StringField()
    data = DictField(default={})
    created_at = DateTimeField(default=datetime.utcnow)

class TagRecord(Document):
    """
    Represents a single tag in a hierarchical structure.
    """
    __collection__ = "tags"

    name = StringField(required=True)
    full_name = StringField(required=True, unique=True)
    parent = ReferenceField('TagRecord', required=False)

    class Meta:
        indexes = [
            [('full_name', 1)]
        ]

    @property
    def fullName(self):
        """
        Compatibility property - returns full_name.
        Matches old fs_db TagRecord.fullName property.
        """
        return getattr(self, 'full_name')

    def child_tags(self):
        """
        Get all child tags of this tag.
        Compatibility method matching old fs_db API.
        """
        return self.__class__.find({'parent': self.pk})

    def tagged_files(self):
        """
        Get all files that have this tag.
        Compatibility method matching old fs_db API.
        """
        # Find all FileRecord instances that have this tag in their tags list
        # or have AnnotationRecord entries linking to this tag
        
        # First approach: files with tag name in tags list
        full_name_val = getattr(self, 'full_name')
        files_with_tag = FileRecord.find({'tags': full_name_val})
        
        # Second approach: files linked via AnnotationRecord
        annotations = AnnotationRecord.find({'tag': self.pk})
        files_from_annotations = [ann.file for ann in annotations]
        
        # Combine and deduplicate
        all_files = list(files_with_tag) + files_from_annotations
        seen_ids = set()
        unique_files = []
        for file in all_files:
            if file.pk not in seen_ids:
                seen_ids.add(file.pk)
                unique_files.append(file)
        
        return unique_files

    def add_to_file_rec(self, file_record):
        """
        Add this tag to a file record.
        Compatibility method matching old fs_db API.
        """
        # Add tag name to file's tags list if not already present
        full_name_val = getattr(self, 'full_name')
        current_tags = file_record.list_get('tags')
        if full_name_val not in current_tags:
            file_record.list_append('tags', full_name_val)

    @classmethod
    def get_tags_of_file(cls, file_record):
        """
        Get all tags associated with a file record.
        Compatibility method matching old fs_db API.
        """
        # Get tags from file's tags list
        file_tags = file_record.list_get('tags')
        tag_objects = []
        
        for tag_name in file_tags:
            tag_obj = cls.find_one({'full_name': tag_name})
            if tag_obj:
                tag_objects.append(tag_obj)
        
        # Also get tags from AnnotationRecord entries
        annotations = AnnotationRecord.find({'file': file_record.pk})
        for ann in annotations:
            if ann.tag not in tag_objects:
                tag_objects.append(ann.tag)
        
        return tag_objects

    @classmethod
    def get_tags_report(cls):
        """
        Generate a report of all tags.
        Compatibility method matching old fs_db API.
        """
        all_tags = cls.find({})
        report = []
        
        for tag in all_tags:
            child_count = len(tag.child_tags())
            file_count = len(tag.tagged_files())
            full_name_val = getattr(tag, 'full_name')
            
            report.append({
                'tag_name': full_name_val,
                'child_tags_count': child_count,
                'tagged_files_count': file_count
            })
        
        # Print report
        print("Tags Report:")
        print("-" * 50)
        for item in report:
            print(f"Tag: {item['tag_name']}")
            print(f"  Child tags: {item['child_tags_count']}")
            print(f"  Tagged files: {item['tagged_files_count']}")
            print()
        
        return report

    def delete_rec(self):
        """
        Delete record from database (wrapper-style compatibility).
        Alias for delete() method to match old fs_db API.
        """
        self.delete()

class FileRecord(CollectionItem):
    """
    Represents a single file in the database.
    """
    __collection__ = "files"

    # Path components
    local_path = StringField(required=True, unique=True)
    name = StringField(required=True)
    ext = StringField()
    
    # File metadata
    size = IntField()
    md5 = StringField()
    metadata = DictField(default={}) # For EXIF, etc.
    
    # Compatibility fields from old fs_db
    source = StringField()  # Source URL
    indexed_by = ListField(StringField(), default=[])  # Track indexing status

    # AI-generated data
    ai_expertise = ListField(EmbeddedDocumentField(AIExpertise), default=[])

    class Meta:
        indexes = [
            [('md5', 1)],
            [('local_path', 1)]
        ]

    @property
    def full_path(self):
        """
        Compatibility property - returns the full local path.
        Matches old fs_db FileRecord.full_path property.
        """
        return self.local_path

    @classmethod
    def get_record_by_path(cls, file_path):
        """
        Get FileRecord by file path.
        Compatibility method matching old fs_db API.
        """
        return cls.find_one({'local_path': file_path})

    @classmethod
    def add_file_records_from_folder(cls, folder_path):
        """
        Add file records from a folder.
        Compatibility method matching old fs_db API.
        """
        import os
        from tqdm import tqdm
        
        added_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in tqdm(files, desc=f"Processing {root}"):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    # Check if already exists
                    existing = cls.get_record_by_path(file_path)
                    if not existing:
                        try:
                            file_stat = os.stat(file_path)
                            name, ext = os.path.splitext(file)
                            
                            file_record = cls.new_record(
                                local_path=file_path,
                                name=name,
                                ext=ext.lower(),
                                size=file_stat.st_size
                            )
                            added_files.append(file_record)
                        except Exception as e:
                            print(f"Error indexing {file_path}: {e}")
        
        return added_files

    def move_to_folder(self, dest_folder):
        """
        Move file to destination folder and update database record.
        Compatibility method matching old fs_db API.
        """
        import os
        import shutil
        
        # Get the actual string value from the field
        current_path = getattr(self, 'local_path')
        
        if not os.path.exists(current_path):
            raise FileNotFoundError(f"Source file not found: {current_path}")
        
        # Create destination directory if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # Build destination path
        filename = os.path.basename(current_path)
        dest_path = os.path.join(dest_folder, filename)
        
        # Move the file
        shutil.move(current_path, dest_path)
        
        # Update database record - this will auto-save
        setattr(self, 'local_path', dest_path)

    def refresh_thumb(self):
        """
        Refresh thumbnail for this file.
        Compatibility method matching old fs_db API.
        """
        # Placeholder implementation - would integrate with thumbnail generation system
        current_path = getattr(self, 'local_path')
        print(f"Refreshing thumbnail for {current_path}")
        # Add actual thumbnail refresh logic here
        pass

    def delete_rec(self):
        """
        Delete record from database (wrapper-style compatibility).
        Alias for delete() method to match old fs_db API.
        """
        self.delete()

class AnnotationRecord(CollectionItem):
    """
    Represents a specific annotation on a file, such as a bounding box
    or a polygon, linked to a specific tag.
    """
    __collection__ = "annotations"

    file = ReferenceField(FileRecord, required=True)
    tag = ReferenceField(TagRecord, required=True)
    
    # Annotation data (e.g., for bounding boxes, polygons)
    annotation_type = StringField(required=True) # 'bbox', 'polygon', etc.
    points = ListField(ListField(FloatField())) # e.g., [[x1, y1], [x2, y2]]
    
    class Meta:
        indexes = [
            [('file', 1)],
            [('tag', 1)]
        ]

class RelationRecord(Document):
    """
    Represents a directional relationship between any two CollectionItems.
    """
    __collection__ = "relations"

    # Using GenericReferenceField to link to any document type
    subject = GenericReferenceField(required=True)
    object = GenericReferenceField(required=True)
    
    relation_type = StringField(required=True)
    
    class Meta:
        indexes = [
            # Compound index to quickly find all relations for a subject
            [('subject.pk', 1), ('subject._cls', 1)],
            # Compound index to quickly find all relations for an object
            [('object.pk', 1), ('object._cls', 1)],
        ]
