import pytest
import asyncio
from src.core.database.models.task import TaskRecord
from src.core.engine.tasks import TaskDispatcher

@pytest.fixture(scope="function")
async def db_teardown_tasks():
    from src.core.database.manager import db_manager
    if db_manager.db is not None:
        await db_manager.db.drop_collection("tasks")
    yield
    if db_manager.db is not None:
        await db_manager.db.drop_collection("tasks")

@pytest.mark.asyncio
async def test_task_persistence(db_teardown_tasks):
    dispatcher = TaskDispatcher()
    task = await dispatcher.submit_task("TEST_JOB", {"foo": "bar"})
    
    assert task.id is not None
    assert task.status == "pending"
    assert task.payload["foo"] == "bar"

    # Reload
    loaded = await TaskRecord.get(task.id)
    assert loaded.status == "pending"

@pytest.mark.asyncio
async def test_task_execution(db_teardown_tasks):
    dispatcher = TaskDispatcher()
    
    # Define a handler
    async def dummy_handler(task: TaskRecord):
        return {"processed": True}
    
    dispatcher.register_handler("TEST_JOB", dummy_handler)
    
    # Submit
    task = await dispatcher.submit_task("TEST_JOB", {})
    
    # Manually trigger process in test (avoid async loop timing issues)
    # In real app, start() runs the loop. Here we invoke execution directly.
    await dispatcher._process_task(task)
    
    # Verify completion
    loaded = await TaskRecord.get(task.id)
    assert loaded.status == "completed"
    assert loaded.result["processed"] is True
    assert loaded.completed_at > 0

@pytest.mark.asyncio
async def test_task_failure(db_teardown_tasks):
    dispatcher = TaskDispatcher()
    
    async def failing_handler(task):
        raise ValueError("Boom!")
        
    dispatcher.register_handler("FAIL_JOB", failing_handler)
    task = await dispatcher.submit_task("FAIL_JOB", {})
    
    await dispatcher._process_task(task)
    
    loaded = await TaskRecord.get(task.id)
    assert loaded.status == "failed"
    assert "Boom!" in loaded.error
