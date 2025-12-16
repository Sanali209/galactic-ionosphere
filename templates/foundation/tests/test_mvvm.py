import pytest
from unittest.mock import MagicMock
from src.ui.mvvm.viewmodel import BaseViewModel
from src.ui.mvvm.provider import ViewModelProvider
from src.ui.viewmodels.main_viewmodel import MainViewModel

def test_base_viewmodel_init():
    locator = MagicMock()
    vm = BaseViewModel(locator)
    assert vm.locator == locator

def test_viewmodel_provider(qapp):
    if qapp is None: pytest.skip("No QApp")
    
    locator = MagicMock()
    provider = ViewModelProvider(locator)
    
    vm1 = provider.get(MainViewModel)
    vm2 = provider.get(MainViewModel)
    
    # Check singleton/caching behavior
    assert isinstance(vm1, MainViewModel)
    assert vm1 is vm2
    assert vm1.locator == locator

def test_main_viewmodel_signals(qapp):
    if qapp is None: pytest.skip("No QApp")
    
    locator = MagicMock()
    vm = MainViewModel(locator)
    
    mock_slot = MagicMock()
    vm.statusMessageChanged.connect(mock_slot)
    
    # Trigger property change
    vm.status_message = "New Status"
    
    mock_slot.assert_called_with("New Status")
    assert vm.status_message == "New Status"

def test_main_viewmodel_log_action(qapp):
    if qapp is None: pytest.skip("No QApp")
    
    locator = MagicMock()
    vm = MainViewModel(locator)
    
    vm.log_action("Test Action")
    
    assert vm.status_message == "Last Action: Test Action"
