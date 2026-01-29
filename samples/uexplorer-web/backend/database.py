"""
Database initialization and configuration for UExplorer Web
"""
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from loguru import logger

from models import (
    FileRecord, DirectoryRecord, Tag, FileTag, Album, FileAlbum,
    DetectionClass, DetectionInstance, Relation, EmbeddingRecord,
    AnnotationJob, AnnotationRecord, Rule, TaskRecord, JournalEvent
)


# Database configuration
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "uexplorer_web"

# Global client
mongo_client = None


async def init_database():
    """Initialize MongoDB and Beanie ODM"""
    global mongo_client
    
    logger.info(f"Connecting to MongoDB: {MONGODB_URL}")
    mongo_client = AsyncIOMotorClient(MONGODB_URL)
    
    # Initialize Beanie with all document models
    await init_beanie(
        database=mongo_client[DATABASE_NAME],
        document_models=[
            FileRecord,
            DirectoryRecord,
            Tag,
            FileTag,
            Album,
            FileAlbum,
            DetectionClass,
            DetectionInstance,
            Relation,
            EmbeddingRecord,
            AnnotationJob,
            AnnotationRecord,
            Rule,
            TaskRecord,
            JournalEvent,
        ]
    )
    
    logger.info(f"✓ Database initialized: {DATABASE_NAME}")
    logger.info("✓ All Beanie models registered")


async def close_database():
    """Close database connection"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("Database connection closed")
