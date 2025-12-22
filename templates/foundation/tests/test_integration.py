"""
Integration Tests for Cross-System Workflows.

Tests interactions between major Foundation systems:
- NodeGraph + UCoreFS
- MVVM Binding + UI 
- ORM + Services
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId

# Pytest fixtures
@pytest.fixture
def mock_locator():
    """Create mock ServiceLocator with systems."""
    locator = MagicMock()
    locator.config = MagicMock()
    return locator


@pytest.fixture
def mock_fs_service():
    """Create mock FSService."""
    service = MagicMock()
    service.search_by_name = AsyncMock(return_value=[])
    service.get_by_path = AsyncMock(return_value=None)
    return service


# =============================================================================
# NodeGraph + UCoreFS Integration
# =============================================================================

class TestNodeGraphUCorefsIntegration:
    """Tests for NodeGraph nodes interacting with UCoreFS."""
    
    @pytest.mark.asyncio
    async def test_file_query_node_returns_results(self, mock_locator, mock_fs_service):
        """Test FileQueryNode returns file IDs from UCoreFS."""
        from src.nodegraph.nodes.ucorefs_nodes import FileQueryNode
        
        # Setup mock
        mock_record = MagicMock()
        mock_record.id = ObjectId()
        mock_fs_service.search_by_name.return_value = [mock_record]
        mock_locator.get_system.return_value = mock_fs_service
        
        # Create and execute node
        node = FileQueryNode()
        node._input_pins["pattern"].default_value = "test.*"
        
        context = {"locator": mock_locator}
        result = await node.execute(context)
        
        assert result["count"] == 1
        assert len(result["files"]) == 1
    
    @pytest.mark.asyncio
    async def test_file_query_node_handles_empty_results(self, mock_locator, mock_fs_service):
        """Test FileQueryNode handles no results gracefully."""
        from src.nodegraph.nodes.ucorefs_nodes import FileQueryNode
        
        mock_fs_service.search_by_name.return_value = []
        mock_locator.get_system.return_value = mock_fs_service
        
        node = FileQueryNode()
        context = {"locator": mock_locator}
        result = await node.execute(context)
        
        assert result["count"] == 0
        assert result["files"] == []
    
    @pytest.mark.asyncio
    async def test_get_file_by_path_found(self, mock_locator, mock_fs_service):
        """Test GetFileByPathNode when file exists."""
        from src.nodegraph.nodes.ucorefs_nodes import GetFileByPathNode
        
        mock_record = MagicMock()
        mock_record.id = ObjectId()
        mock_fs_service.get_by_path.return_value = mock_record
        mock_locator.get_system.return_value = mock_fs_service
        
        node = GetFileByPathNode()
        node._input_pins["path"].default_value = "/test/file.txt"
        
        context = {"locator": mock_locator}
        result = await node.execute(context)
        
        assert result["found"] is True
        assert result["file_id"] is not None
    
    @pytest.mark.asyncio
    async def test_get_file_by_path_not_found(self, mock_locator, mock_fs_service):
        """Test GetFileByPathNode when file doesn't exist."""
        from src.nodegraph.nodes.ucorefs_nodes import GetFileByPathNode
        
        mock_fs_service.get_by_path.return_value = None
        mock_locator.get_system.return_value = mock_fs_service
        
        node = GetFileByPathNode()
        node._input_pins["path"].default_value = "/nonexistent/file.txt"
        
        context = {"locator": mock_locator}
        result = await node.execute(context)
        
        assert result["found"] is False


# =============================================================================
# MVVM Binding + ViewModel Integration
# =============================================================================

class TestMVVMIntegration:
    """Tests for MVVM binding system integration."""
    
    def test_viewmodel_binds_to_widget(self, qapp):
        """Test ViewModel property binds to widget correctly."""
        from PySide6.QtWidgets import QLabel
        from src.ui.mvvm.bindable import BindableBase, BindableProperty
        from src.ui.mvvm.binding import bind, BindingMode
        from PySide6.QtCore import Signal
        
        class TestVM(BindableBase):
            textChanged = Signal(str)
            text = BindableProperty(default="initial")
        
        vm = TestVM()
        label = QLabel()
        
        bind(vm, "text", label, "text")
        
        assert label.text() == "initial"
        
        vm.text = "updated"
        assert label.text() == "updated"
    
    def test_viewmodel_provider_creates_vm(self, qapp):
        """Test ViewModelProvider creates and caches ViewModels."""
        from src.ui.mvvm.provider import ViewModelProvider
        from src.ui.mvvm.bindable import BindableBase
        
        class TestVM(BindableBase):
            pass
        
        mock_locator = MagicMock()
        provider = ViewModelProvider(mock_locator)
        
        vm1 = provider.get(TestVM)
        vm2 = provider.get(TestVM)
        
        assert vm1 is vm2  # Same instance (cached)


# =============================================================================
# ServiceLocator + Systems Integration
# =============================================================================

class TestServiceLocatorIntegration:
    """Tests for ServiceLocator system management."""
    
    @pytest.mark.asyncio
    async def test_systems_start_in_order(self):
        """Test that systems are started in registration order."""
        from src.core.locator import ServiceLocator
        from src.core.base_system import BaseSystem
        
        start_order = []
        
        class SystemA(BaseSystem):
            async def initialize(self):
                start_order.append("A")
                await super().initialize()
            async def shutdown(self):
                await super().shutdown()
        
        class SystemB(BaseSystem):
            async def initialize(self):
                start_order.append("B")
                await super().initialize()
            async def shutdown(self):
                await super().shutdown()
        
        sl = ServiceLocator.__new__(ServiceLocator)
        sl._instance = None
        sl._initialized = False
        sl.init("config.json")
        
        sl.register_system(SystemA)
        sl.register_system(SystemB)
        
        await sl.start_all()
        
        assert start_order == ["A", "B"]
    
    @pytest.mark.asyncio
    async def test_systems_stop_in_reverse_order(self):
        """Test that systems are stopped in reverse order."""
        from src.core.locator import ServiceLocator
        from src.core.base_system import BaseSystem
        
        stop_order = []
        
        class SystemA(BaseSystem):
            async def initialize(self):
                await super().initialize()
            async def shutdown(self):
                stop_order.append("A")
                await super().shutdown()
        
        class SystemB(BaseSystem):
            async def initialize(self):
                await super().initialize()
            async def shutdown(self):
                stop_order.append("B")
                await super().shutdown()
        
        sl = ServiceLocator.__new__(ServiceLocator)
        sl._instance = None
        sl._initialized = False
        sl.init("config.json")
        
        sl.register_system(SystemA)
        sl.register_system(SystemB)
        
        await sl.start_all()
        await sl.stop_all()
        
        assert stop_order == ["B", "A"]  # Reverse order


# Pytest fixture for QApplication
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
