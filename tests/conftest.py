import pytest
import asyncio
from src.core.locator import sl
from src.core.database.manager import db_manager

@pytest.fixture(scope="function", autouse=True)
def initialize_core():
    """Initialize the Service Locator for each test to ensure fresh loop reference"""
    sl.init("settings.json") 
    db_manager.init()
    yield
    # Teardown if needed

@pytest.fixture(scope="function")
async def db_teardown():
    """Cleanup database after each test"""
    # Clear collections used in tests
    if db_manager.db is not None:
        await db_manager.db.drop_collection("test_users")
        await db_manager.db.drop_collection("test_valid")
        await db_manager.db.drop_collection("products")
        # App Collections
        await db_manager.db.drop_collection("gallery_entities")
        await db_manager.db.drop_collection("tags")
        await db_manager.db.drop_collection("tasks")
        await db_manager.db.drop_collection("references")
    yield
    if db_manager.db is not None:
        await db_manager.db.drop_collection("test_users")
        await db_manager.db.drop_collection("test_valid")
        await db_manager.db.drop_collection("products")
        await db_manager.db.drop_collection("gallery_entities")
        await db_manager.db.drop_collection("tags")
        await db_manager.db.drop_collection("tasks")
        await db_manager.db.drop_collection("references")
