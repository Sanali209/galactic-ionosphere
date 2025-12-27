from loguru import logger
from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.mongoODM.db_component import MongoODMComponent
from .services import IndexingService, TagService, AnnotationService, RelationService

def create_files_db_service(config: Config, message_bus: MessageBus):
    """
    Factory function to initialize and return all components related to the files_db.
    """
    # 1. Initialize the core ODM component
    odm_component = MongoODMComponent(config=config, message_bus=message_bus)

    # 2. Initialize all service components
    indexing_service = IndexingService(message_bus=message_bus)
    tag_service = TagService(message_bus=message_bus)
    annotation_service = AnnotationService(message_bus=message_bus)
    relation_service = RelationService(message_bus=message_bus)

    logger.info("Initialized all files_db components and services.")

    # Return a list of all components to be managed by the application
    return [
        odm_component,
        indexing_service,
        tag_service,
        annotation_service,
        relation_service
    ]
