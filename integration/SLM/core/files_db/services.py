import os
from typing import Optional
from loguru import logger
from SLM.core.component import Component
from SLM.core.message_bus import MessageBus
from .odm_models import FileRecord, TagRecord, AnnotationRecord, RelationRecord
from SLM.core.mongoODM.documents import Document

class IndexingService(Component):
    """
    A service component responsible for indexing files and creating FileRecord documents.
    """
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.subscribe("files.index_file", self.index_file)

    def start(self):
        logger.info("IndexingService started.")

    def stop(self):
        logger.info("IndexingService stopped.")

    def index_file(self, msg_type, file_path: str, metadata: Optional[dict] = None):
        """
        Creates or updates a FileRecord for a given file path.
        """
        assert self.message_bus is not None
        try:
            # Check if a record with this path already exists
            assert FileRecord.objects is not None
            existing_record = FileRecord.objects.find_one({'local_path': file_path})

            if existing_record:
                logger.info(f"File already indexed: {file_path}")
                return existing_record

            # Get file stats
            file_stat = os.stat(file_path)
            file_name, file_ext = os.path.splitext(os.path.basename(file_path))

            # Create a new record - automatically saves to database
            new_record = FileRecord.new_record(**{
                'local_path': file_path,
                'name': file_name,
                'ext': file_ext,
                'size': file_stat.st_size,
                'metadata': metadata or {}
            })
            
            logger.info(f"Successfully indexed file: {file_path}")
            self.message_bus.publish("files.file_indexed", record=new_record)
            
            return new_record

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            self.message_bus.publish("files.index_error", path=file_path, error=str(e))
            return None

class TagService(Component):
    """
    A service component for managing tags.
    """
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.subscribe("tags.create_tag", self.get_or_create_tag)

    def start(self):
        logger.info("TagService started.")

    def stop(self):
        logger.info("TagService stopped.")

    def get_or_create_tag(self, msg_type, full_tag_name: str):
        """
        Gets or creates a tag, including its parent tags if they don't exist.
        Example: "person/head/hair"
        """
        assert self.message_bus is not None
        assert TagRecord.objects is not None
        
        try:
            # Check if the tag already exists
            existing_tag = TagRecord.objects.find_one({'full_name': full_tag_name})
            if existing_tag:
                return existing_tag

            # If not, create it and its parents recursively
            parts = full_tag_name.split('/')
            parent_doc = None
            created_tag = None

            for i, part in enumerate(parts):
                current_full_name = "/".join(parts[:i+1])
                
                tag_doc = TagRecord.objects.find_one({'full_name': current_full_name})
                if not tag_doc:
                    # Create new tag record - auto-saves to database
                    tag_doc = TagRecord.new_record(**{
                        'name': part, 
                        'full_name': current_full_name, 
                        'parent': parent_doc
                    })
                
                parent_doc = tag_doc
                created_tag = tag_doc

            logger.info(f"Successfully created tag: {full_tag_name}")
            self.message_bus.publish("tags.tag_created", tag=created_tag)
            return created_tag

        except Exception as e:
            logger.error(f"Error creating tag {full_tag_name}: {e}")
            self.message_bus.publish("tags.create_error", name=full_tag_name, error=str(e))
            return None

class AnnotationService(Component):
    """
    A service component for managing annotations.
    """
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.subscribe("annotations.create", self.create_annotation)

    def start(self):
        logger.info("AnnotationService started.")

    def stop(self):
        logger.info("AnnotationService stopped.")

    def create_annotation(self, msg_type, file: FileRecord, tag: TagRecord, annotation_type: str, points: list):
        """
        Creates an annotation for a given file and tag.
        """
        assert self.message_bus is not None
        assert AnnotationRecord.objects is not None
        
        try:
            # Create new annotation record - auto-saves to database
            annotation = AnnotationRecord.new_record(**{
                'file': file,
                'tag': tag,
                'annotation_type': annotation_type,
                'points': points
            })
            
            logger.info(f"Successfully created annotation for file {file.pk} with tag {tag.pk}")
            self.message_bus.publish("annotations.created", annotation=annotation)
            return annotation

        except Exception as e:
            logger.error(f"Error creating annotation: {e}")
            self.message_bus.publish("annotations.create_error", error=str(e))
            return None

class RelationService(Component):
    """
    A service component for managing relationships between documents.
    """
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.subscribe("relations.create", self.create_relation)

    def start(self):
        logger.info("RelationService started.")

    def stop(self):
        logger.info("RelationService stopped.")

    def create_relation(self, msg_type, subject: Document, object: Document, relation_type: str):
        """
        Creates a directional relationship between two documents.
        """
        assert self.message_bus is not None
        assert RelationRecord.objects is not None
        
        try:
            # Create new relation record - auto-saves to database  
            relation = RelationRecord.new_record(**{
                'subject': subject,
                'object': object,
                'relation_type': relation_type
            })
            
            logger.info(f"Successfully created '{relation_type}' relation between {subject} and {object}")
            self.message_bus.publish("relations.created", relation=relation)
            return relation

        except Exception as e:
            logger.error(f"Error creating relation: {e}")
            self.message_bus.publish("relations.create_error", error=str(e))
            return None
