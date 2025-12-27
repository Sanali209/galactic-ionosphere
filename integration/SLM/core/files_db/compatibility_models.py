"""
Additional models for compatibility with old fs_db system.
These models extend the core files_db functionality to match old API.
"""

from datetime import datetime
from SLM.core.mongoODM.documents import Document, EmbeddedDocument
from SLM.core.mongoODM.fields import (
    StringField,
    IntField,
    ListField,
    DictField,
    BooleanField,
    ReferenceField,
    DateTimeField
)
from .odm_models import FileRecord, TagRecord


class AnnotationJob(Document):
    """
    Manages annotation workflows for machine learning datasets.
    Compatibility with old fs_db AnnotationJob functionality.
    """
    __collection__ = "annotation_jobs"
    
    name = StringField(required=True, unique=True)
    description = StringField()
    choices = ListField(StringField(), default=[])  # Available annotation labels
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    @classmethod
    def get_by_name(cls, job_name):
        """Get annotation job by name - compatibility method."""
        return cls.find_one({'name': job_name})
    
    def annotate_file(self, file_record, label):
        """
        Annotate a file with a label for this job.
        Compatibility method matching old fs_db API.
        """
        if label not in self.choices:
            raise ValueError(f"Label '{label}' not in job choices: {self.choices}")
        
        # Create or update annotation record
        existing = AnnotationJobRecord.find_one({
            'job': self.pk,
            'file': file_record.pk
        })
        
        if existing:
            existing.label = label
        else:
            AnnotationJobRecord.new_record(
                job=self,
                file=file_record,
                label=label
            )
    
    def remove_annotation_record(self, file_record):
        """Remove annotation for a file from this job."""
        existing = AnnotationJobRecord.find_one({
            'job': self.pk,
            'file': file_record.pk
        })
        if existing:
            existing.delete()
    
    def get_ann_records_by_label(self, label):
        """Get all annotation records for a specific label."""
        return AnnotationJobRecord.find({
            'job': self.pk,
            'label': label
        })
    
    def add_annotation_choices(self, new_choices):
        """Add new choices to the annotation job."""
        current_choices = self.list_get('choices')
        for choice in new_choices:
            if choice not in current_choices:
                self.list_append('choices', choice)
    
    def rename_annotation_label(self, old_label, new_label):
        """Rename an annotation label across all records."""
        # Update choices list
        choices = self.list_get('choices')
        if old_label in choices:
            choices[choices.index(old_label)] = new_label
            self.choices = choices
        
        # Update all annotation records
        records = AnnotationJobRecord.find({
            'job': self.pk,
            'label': old_label
        })
        for record in records:
            if record is not None:
                record.label = new_label
    
    def clear_job(self):
        """Clear all annotations for this job."""
        records = AnnotationJobRecord.find({'job': self.pk})
        for record in records:
            if record is not None:
                record.delete()
    
    def remove_annotation_dublicates2(self):
        """Remove duplicate annotation records."""
        # Group by file
        file_records = {}
        records = AnnotationJobRecord.find({'job': self.pk})
        
        for record in records:
            if record is not None:
                file_id = record.file.pk if record.file else None
                if file_id:
                    if file_id not in file_records:
                        file_records[file_id] = []
                    file_records[file_id].append(record)
        
        # Remove duplicates, keep the latest
        for file_id, records_list in file_records.items():
            if len(records_list) > 1:
                # Sort by created_at, keep the latest
                records_list.sort(key=lambda x: getattr(x, 'created_at', datetime.min))
                for record in records_list[:-1]:  # Remove all but the last
                    record.delete()
    
    def remove_broken_annotations(self):
        """Remove annotations pointing to non-existent files."""
        records = AnnotationJobRecord.find({'job': self.pk})
        for record in records:
            if record is not None and (not record.file or not record.file.pk):
                record.delete()


class AnnotationJobRecord(Document):
    """
    Individual annotation record linking a file to a label in an annotation job.
    """
    __collection__ = "annotation_job_records"
    
    job = ReferenceField(AnnotationJob, required=True)
    file = ReferenceField(FileRecord, required=True) 
    label = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    
    class Meta:
        indexes = [
            [('job', 1), ('file', 1)],  # Compound index for uniqueness
            [('job', 1), ('label', 1)]  # For querying by label
        ]


class Detection(Document):
    """
    Object detection results for images.
    Compatibility with old fs_db Detection functionality.
    """
    __collection__ = "detections"
    
    parent_image_id = ReferenceField(FileRecord, required=True)
    detection_type = StringField(default="face_detection")
    confidence = IntField(default=0)  # 0-100
    bbox = ListField(IntField(), default=[])  # [x, y, width, height]
    metadata = DictField(default={})
    created_at = DateTimeField(default=datetime.utcnow)
    
    class Meta:
        indexes = [
            [('parent_image_id', 1)],
            [('detection_type', 1)]
        ]
    
    def delete_rec(self):
        """Delete record - compatibility method."""
        self.delete()


# Add delete_rec method to RelationRecord for compatibility
def add_delete_rec_to_relation():
    """Add delete_rec method to RelationRecord for compatibility."""
    from .odm_models import RelationRecord
    
    def delete_rec(self):
        """Delete record - compatibility method."""
        self.delete()
    
    # Add method to RelationRecord class
    RelationRecord.delete_rec = delete_rec

# Call the function to add the method
add_delete_rec_to_relation()
