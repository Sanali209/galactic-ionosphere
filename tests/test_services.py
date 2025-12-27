import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.locator import ServiceLocator
from src.core.assets.manager import AssetManager
from src.core.journal.service import JournalService
from src.core.base_system import BaseSystem

# --- Locator Tests ---
class DummySystem(BaseSystem):
    async def initialize(self):
        await super().initialize()
    async def shutdown(self):
        await super().shutdown()

@pytest.mark.asyncio
async def test_locator_registration():
    # Reset Singleton for test
    ServiceLocator._instance = None
    sl = ServiceLocator()
    sl.init() # Ensure init is called
    
    sl.register_system(DummySystem)
    
    sys = sl.get_system(DummySystem)
    assert isinstance(sys, DummySystem)
    assert sys.locator == sl

# --- Asset Manager Tests ---
@pytest.mark.asyncio
async def test_asset_ingest_no_handler():
    # Setup
    sl = MagicMock()
    config = MagicMock()
    assets = AssetManager(sl, config)
    
    result = await assets.ingest("somefile.unknown")
    
    # Assert
    # AssetManager currently returns a dict even if no handler (placeholder)
    assert result == {"path": "somefile.unknown", "status": "ingested"}

# --- Journal Tests ---
@pytest.mark.asyncio
async def test_journal_log():
    sl = MagicMock()
    config = MagicMock()
    journal = JournalService(sl, config)
    
    # Mock JournalEntry.save to avoid DB call
    # IMPORTANT: Need to mock the IMPORTED class in the service module
    with patch('src.core.journal.service.JournalEntry') as MockEntry:
        mock_instance = MockEntry.return_value
        mock_instance.save = AsyncMock()
        
        await journal.log("INFO", "Test", "Message")
        
        mock_instance.save.assert_called_once()
