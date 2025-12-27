"""
Test Navigation Service logic.
"""
import pytest
from unittest.mock import MagicMock, Mock
from src.ui.navigation.service import NavigationService, NavigationContext, NavigationHandler

class MockHandler(NavigationHandler):
    def __init__(self, name, priority=0, handles_str=False):
        self._name = name
        self._priority = priority
        self._handles_str = handles_str
        self.handled_existing = []
        self.handled_new = []
        self.owned_targets = set()
        
    @property
    def priority(self):
        return self._priority
        
    def can_handle(self, data):
        if self._handles_str and isinstance(data, str):
            return True
        return False
        
    def owns_target(self, target_id):
        return target_id in self.owned_targets
        
    def handle_existing(self, target_id, data, context):
        self.handled_existing.append((target_id, data))
        
    def handle_new(self, data, context):
        self.handled_new.append(data)
        
    def __repr__(self):
        return f"MockHandler({self._name})"

@pytest.fixture
def service_locator():
    mock = MagicMock()
    mock.get_system = MagicMock(return_value=None)
    return mock

@pytest.fixture
def docking_service():
    mock = MagicMock()
    mock.get_active_document_id = MagicMock(return_value=None)
    return mock

@pytest.mark.asyncio
async def test_navigation_registration(service_locator):
    service = NavigationService(service_locator, {})
    await service.initialize()
    
    handler1 = MockHandler("h1", priority=10)
    handler2 = MockHandler("h2", priority=20)
    
    service.register_handler(handler1)
    service.register_handler(handler2)
    
    # Handlers should be sorted by priority (h2 first)
    assert service._handlers[0] == handler2
    assert service._handlers[1] == handler1

@pytest.mark.asyncio
async def test_navigate_new(service_locator):
    service = NavigationService(service_locator, {})
    await service.initialize()
    
    handler = MockHandler("h1", handles_str=True)
    service.register_handler(handler)
    
    # Test navigating to data that requires new view (no context)
    result = service.navigate("path/to/data")
    
    assert result is True
    assert len(handler.handled_new) == 1
    assert handler.handled_new[0] == "path/to/data"

@pytest.mark.asyncio
async def test_navigate_existing_explicit_target(service_locator):
    service = NavigationService(service_locator, {})
    await service.initialize()
    
    handler = MockHandler("h1", handles_str=True)
    handler.owned_targets.add("doc_1")
    service.register_handler(handler)
    
    # Test navigating to explicit target
    ctx = NavigationContext(target_id="doc_1")
    result = service.navigate("data", ctx)
    
    assert result is True
    assert len(handler.handled_existing) == 1
    assert handler.handled_existing[0] == ("doc_1", "data")

@pytest.mark.asyncio
async def test_navigate_active_target(service_locator, docking_service):
    # Setup locator to return docking service
    service_locator.get_system = MagicMock(return_value=docking_service)
    
    service = NavigationService(service_locator, {})
    await service.initialize()
    
    handler = MockHandler("h1", handles_str=True)
    handler.owned_targets.add("doc_active")
    service.register_handler(handler)
    
    # Mock active document
    docking_service.get_active_document_id.return_value = "doc_active"
    
    # Test navigate without explicit target -> should use active
    result = service.navigate("data")
    
    assert result is True
    assert len(handler.handled_existing) == 1
    assert handler.handled_existing[0] == ("doc_active", "data")
    
@pytest.mark.asyncio
async def test_navigate_active_target_mismatch(service_locator, docking_service):
    # Setup locator to return docking service
    service_locator.get_system = MagicMock(return_value=docking_service)
    
    service = NavigationService(service_locator, {})
    await service.initialize()
    
    handler = MockHandler("h1", handles_str=True)
    # Handler owns 'doc_inactive', but 'doc_active' is focused and NOT owned by handler
    handler.owned_targets.add("doc_inactive") 
    service.register_handler(handler)
    
    docking_service.get_active_document_id.return_value = "doc_other_type"
    
    # Navigate -> Active doc "doc_other_type" is not owned by handler.
    # Should fall back to creating new (handled_new) 
    # OR find compatible (not impl).
    
    result = service.navigate("data")
    
    assert result is True
    # Should have called handle_new because active target wasn't compatible
    assert len(handler.handled_new) == 1
    assert len(handler.handled_existing) == 0
