import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.core.ai.handlers import AITaskHandlers
from src.core.database.models.task import TaskRecord
from src.core.database.models.image import ImageRecord

@pytest.fixture
def mock_dependencies():
    v_driver = MagicMock()
    v_driver.upsert_vector = AsyncMock()
    
    e_service = MagicMock()
    e_service.load = AsyncMock()
    e_service.encode_image = AsyncMock(return_value=[0.1] * 512)
    
    return v_driver, e_service

@pytest.mark.asyncio
async def test_generate_vectors_flow(db_teardown, mock_dependencies):
    v_driver, e_service = mock_dependencies
    
    # Setup Data
    img = ImageRecord(path="/tmp", filename="ai_test.jpg")
    await img.save()
    
    task = TaskRecord(task_type="GENERATE_VECTORS", payload={"image_id": str(img.id)})
    
    # Handler
    handlers = AITaskHandlers(v_driver, e_service)
    
    # Execution
    result = await handlers.handle_generate_vectors(task)
    
    # Verification
    assert result["vector_len"] == 512
    e_service.encode_image.assert_called()
    v_driver.upsert_vector.assert_called()
    
    # Check payload content passed to Qdrant
    call_args = v_driver.upsert_vector.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert kwargs['point_id'] is not None
    # ObjectId to UUID conversion happens
