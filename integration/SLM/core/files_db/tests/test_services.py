import pytest
import pytest_asyncio
import os
import sys
import asyncio
from mongomock import MongoClient as MockMongoClient

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.files_db.files_db_module import create_files_db_service
from SLM.core.files_db.odm_models import FileRecord, TagRecord, AnnotationRecord, RelationRecord

@pytest.fixture
def mock_config():
    """Fixture to create a mock Config object for testing."""
    conf = Config()
    conf.set_value("mongodb", {
        "host": "localhost",
        "port": 27017,
        "db_name": "test_files_db"
    })
    return conf

@pytest.fixture
def mock_mongo_client():
    """Fixture to create a mock MongoClient."""
    return MockMongoClient()

@pytest_asyncio.fixture
async def service_components(mock_config, mock_mongo_client):
    """
    Fixture to set up all service components for testing, using mongomock.
    """
    message_bus = MessageBus()
    
    # Create all components using the factory
    all_components = create_files_db_service(mock_config, message_bus)
    
    # The first component is the MongoODMComponent, let's mock its client
    odm_component = all_components[0]
    
    # Manually replace the real MongoClient with the mock one
    odm_component.client = mock_mongo_client
    odm_component.db = mock_mongo_client[mock_config.get("mongodb")["db_name"]]

    # Start the component and the message bus processing
    odm_component.start()
    processing_task = await message_bus.start_async_processing_task()

    yield message_bus, all_components

    # Teardown
    message_bus.stop_async_processing()
    if processing_task:
        processing_task.cancel()
    odm_component.stop()

def test_placeholder():
    """A placeholder test to ensure the setup is correct."""
    assert True

# --- Service Tests ---

@pytest.mark.asyncio
async def test_indexing_service(service_components):
    """Test the IndexingService."""
    message_bus, _ = service_components
    
    # Create a dummy file to index
    dummy_file_path = "test_file.txt"
    with open(dummy_file_path, "w") as f:
        f.write("test content")

    # Publish a message to index the file
    await message_bus.publish_async("files.index_file", file_path=dummy_file_path)
    await asyncio.sleep(0.1)  # Allow time for async processing

    # Verify the file was indexed
    assert FileRecord.objects is not None
    loop = asyncio.get_running_loop()
    indexed_file = await loop.run_in_executor(None, FileRecord.objects.find_one, {"local_path": dummy_file_path})
    assert indexed_file is not None
    assert indexed_file.name == "test_file"
    assert indexed_file.ext == ".txt"

    # Clean up the dummy file
    os.remove(dummy_file_path)

@pytest.mark.asyncio
async def test_tag_service_hierarchical(service_components):
    """Test the TagService with hierarchical tags."""
    message_bus, _ = service_components

    # Publish a message to create a hierarchical tag
    full_tag_name = "animal/mammal/cat"
    await message_bus.publish_async("tags.create_tag", full_tag_name=full_tag_name)
    await asyncio.sleep(0.1)  # Allow time for async processing

    # Verify all parts of the hierarchy were created
    assert TagRecord.objects is not None
    loop = asyncio.get_running_loop()
    cat_tag = await loop.run_in_executor(None, TagRecord.objects.find_one, {"full_name": "animal/mammal/cat"})
    mammal_tag = await loop.run_in_executor(None, TagRecord.objects.find_one, {"full_name": "animal/mammal"})
    animal_tag = await loop.run_in_executor(None, TagRecord.objects.find_one, {"full_name": "animal"})

    assert cat_tag is not None
    assert mammal_tag is not None
    assert animal_tag is not None

    assert cat_tag.parent.pk == mammal_tag.pk
    assert mammal_tag.parent.pk == animal_tag.pk
    assert animal_tag.parent is None

@pytest.mark.asyncio
async def test_annotation_and_relation_services(service_components):
    """Test the AnnotationService and RelationService."""
    message_bus, _ = service_components

    # 1. Create a file and a tag to work with
    dummy_file_path = "annotation_test.txt"
    with open(dummy_file_path, "w") as f:
        f.write("content")
    
    await message_bus.publish_async("files.index_file", file_path=dummy_file_path)
    await message_bus.publish_async("tags.create_tag", full_tag_name="object/person")
    await asyncio.sleep(0.1)  # Allow time for async processing

    assert FileRecord.objects is not None
    assert TagRecord.objects is not None
    loop = asyncio.get_running_loop()
    file_doc = await loop.run_in_executor(None, FileRecord.objects.find_one, {"local_path": dummy_file_path})
    tag_doc = await loop.run_in_executor(None, TagRecord.objects.find_one, {"full_name": "object/person"})
    
    assert file_doc is not None
    assert tag_doc is not None

    # 2. Test AnnotationService
    points = [[0.1, 0.1], [0.5, 0.5]]
    await message_bus.publish_async("annotations.create", file=file_doc, tag=tag_doc, annotation_type="bbox", points=points)
    await asyncio.sleep(0.1)  # Allow time for async processing

    assert AnnotationRecord.objects is not None
    annotation = await loop.run_in_executor(None, AnnotationRecord.objects.find_one, {})
    assert annotation is not None
    assert annotation.file.pk == file_doc.pk
    assert annotation.tag.pk == tag_doc.pk
    assert annotation.annotation_type == "bbox"

    # 3. Test RelationService
    await message_bus.publish_async("relations.create", subject=file_doc, object=tag_doc, relation_type="contains")
    await asyncio.sleep(0.1)  # Allow time for async processing

    assert RelationRecord.objects is not None
    relation = await loop.run_in_executor(None, RelationRecord.objects.find_one, {})
    assert relation is not None
    assert relation.relation_type == "contains"
    
    # Verify the generic references
    assert relation.subject.pk == file_doc.pk
    assert isinstance(relation.subject, FileRecord)
    assert relation.object.pk == tag_doc.pk
    assert isinstance(relation.object, TagRecord)

    # Clean up
    os.remove(dummy_file_path)

@pytest.mark.asyncio
async def test_tagging_and_retrieval(service_components):
    """Test creating a tag and associating it with a file."""
    message_bus, _ = service_components
    
    # 1. Index a file
    dummy_file_path = "tagging_test.txt"
    with open(dummy_file_path, "w") as f:
        f.write("content")
    await message_bus.publish_async("files.index_file", file_path=dummy_file_path)
    
    # 2. Create a tag
    await message_bus.publish_async("tags.create_tag", full_tag_name="test/tag")
    await asyncio.sleep(0.1)  # Allow time for async processing

    # 3. Retrieve documents
    assert FileRecord.objects is not None
    assert TagRecord.objects is not None
    loop = asyncio.get_running_loop()
    file_doc = await loop.run_in_executor(None, FileRecord.objects.find_one, {"local_path": dummy_file_path})
    tag_doc = await loop.run_in_executor(None, TagRecord.objects.find_one, {"full_name": "test/tag"})
    assert file_doc is not None
    assert tag_doc is not None

    # 4. Create a relationship to represent tagging
    await message_bus.publish_async("relations.create", subject=file_doc, object=tag_doc, relation_type="has_tag")
    await asyncio.sleep(0.1)  # Allow time for async processing

    # 5. Verify the relationship
    assert RelationRecord.objects is not None
    relation = await loop.run_in_executor(None, RelationRecord.objects.find_one, {"subject._id": file_doc.pk, "object._id": tag_doc.pk})
    assert relation is not None
    assert relation.relation_type == "has_tag"

    os.remove(dummy_file_path)

@pytest.mark.asyncio
async def test_remove_duplicate_file_records(service_components):
    """Test finding and removing duplicate file records."""
    message_bus, _ = service_components
    
    dummy_file_path = "duplicate_test.txt"
    with open(dummy_file_path, "w") as f:
        f.write("content")

    # Index the same file twice
    await message_bus.publish_async("files.index_file", file_path=dummy_file_path)
    await message_bus.publish_async("files.index_file", file_path=dummy_file_path)
    await asyncio.sleep(0.1)  # Allow time for async processing

    # The service logic should prevent exact duplicates, but we can test finding them
    assert FileRecord.objects is not None
    
    # In the new architecture, the IndexingService prevents duplicates, 
    # so we expect only one record.
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(None, list, FileRecord.objects.find({"local_path": dummy_file_path}))
    assert len(records) == 1

    os.remove(dummy_file_path)
