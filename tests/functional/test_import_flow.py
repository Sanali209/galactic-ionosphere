import pytest
import os
import asyncio
from src.core.engine.importer import ImportService
from src.core.database.models.image import ImageRecord

@pytest.fixture
def test_image_file(tmp_path):
    # Create valid dummy jpg
    p = tmp_path / "test_import.jpg"
    p.write_bytes(b"fake_image_content_1234")
    return str(p)

@pytest.mark.asyncio
async def test_import_single_file(db_teardown, test_image_file):
    from src.core.engine.tasks import TaskDispatcher
    from src.core.database.models.task import TaskRecord
    
    dispatcher = TaskDispatcher()
    importer = ImportService(dispatcher=dispatcher)
    
    # We need to mock FileHandlerFactory or ensure JpgHandler doesn't crash on fake bytes.
    # Our JpgHandler stub does nothing, which is fine for now.
    
    record = await importer.process_file(test_image_file)
    
    assert record is not None
    assert record.filename == "test_import.jpg"
    
    # Verify Task Created
    tasks = await TaskRecord.find({"task_type": "GENERATE_THUMBNAIL"})
    assert len(tasks) == 1
    assert tasks[0].payload["content_hash"] == record.content_md5
    
    # Verify DB
    saved = await ImageRecord.get(record.id)
    # saved.path is Directory. test_image_file is Full Path.
    # saved.path = C:/.../pytest-0/...
    # test_image_file = C:\...\pytest-0\...\test.jpg
    
    # Normalize
    saved_path_norm = saved.path.replace("\\", "/")
    test_path_norm = test_image_file.replace("\\", "/")
    
    assert saved_path_norm in test_path_norm
    
@pytest.mark.asyncio
async def test_import_duplicate(db_teardown, test_image_file):
    importer = ImportService()
    
    rec1 = await importer.process_file(test_image_file)
    rec2 = await importer.process_file(test_image_file)
    
    assert rec1.id == rec2.id
    # Should count 1 in DB
    count = await ImageRecord.get_collection().count_documents({})
    assert count == 1
