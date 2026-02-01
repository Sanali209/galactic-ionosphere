import pytest
from PySide6.QtCore import QObject
from src.ui.mvvm.bindable import BindableProperty, BindableBase
from src.ui.mvvm.viewmodel import BaseViewModel
from src.ui.mvvm.sync_manager import ContextSyncManager

class MockLocator:
    """Simple service locator mock for testing."""
    def __init__(self):
        self._systems = {}
    
    def register_system(self, cls, instance):
        self._systems[cls] = instance
        
    def get_system(self, cls):
        return self._systems.get(cls)

class TestViewModel(BaseViewModel):
    """ViewModel for testing basic sync."""
    name = BindableProperty(default="", sync_channel="name_sync")
    count = BindableProperty(default=0)  # No sync_channel

def test_property_synchronization():
    """Test that properties with sync_channel are synchronized across ViewModels."""
    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = TestViewModel(locator)
    vm2 = TestViewModel(locator)
    
    # Initialize reactivity (registers with sync_mgr)
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    # Change vm1.name -> Should update vm2.name
    vm1.name = "Alice"
    assert vm2.name == "Alice"
    
    # Change vm2.name -> Should update vm1.name
    vm2.name = "Bob"
    assert vm1.name == "Bob"
    
    # Change vm1.count -> Should NOT update vm2.count (no sync_channel)
    vm1.count = 10
    assert vm2.count == 0

def test_circular_sync_prevention():
    """Test that synchronization does not cause infinite recursion."""
    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = TestViewModel(locator)
    vm2 = TestViewModel(locator)
    
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    # If we get here without a RecursionError, the _mvvm_internal_update guard is working
    vm1.name = "Test"
    assert vm2.name == "Test"

def test_mapper_translation():
    """Test that signal value translation works via mapper."""
    class TranslatedViewModel(BaseViewModel):
        # Maps integer value to string "Value: {x}"
        display_val = BindableProperty(
            default=0, 
            sync_channel="val_sync", 
            mapper=lambda x: f"Value: {x}"
        )
        
        # Receives the string
        received_val = BindableProperty(default="", sync_channel="val_sync")

    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = TranslatedViewModel(locator)
    vm2 = TranslatedViewModel(locator)
    
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    vm1.display_val = 42
    # vm1.display_val internally is 42
    assert vm1.display_val == 42
    
    # vm2 receives "Value: 42" via the val_sync channel
    assert vm2.received_val == "Value: 42"


from src.ui.mvvm.bindable import BindableList, BindableDict

def test_collection_synchronization():
    """Test that mutating a BindableList or BindableDict triggers sync."""
    class CollectionVM(BaseViewModel):
        items = BindableProperty(default=None, sync_channel="items")

    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = CollectionVM(locator)
    vm2 = CollectionVM(locator)
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    # Initialize with BindableList
    list_inst = BindableList(["a", "b"])
    vm1.items = list_inst
    
    # vm2 should now have the same list instance
    assert vm2.items == ["a", "b"]
    assert vm2.items is vm1.items
    
    # Mutate vm1.items
    vm1.items.append("c")
    
    # vm2 should see it (shared instance)
    assert vm2.items == ["a", "b", "c"]
    
    # Verify vm2 emitted notification (SyncManager calls notify_property_changed)
    import unittest.mock as mock
    vm2.notify_property_changed = mock.Mock()
    
    vm1.items.pop()
    assert vm1.items == ["a", "b"]
    assert vm2.items == ["a", "b"]
    
    # Check if vm2 was notified of the change in its 'items' property
    vm2.notify_property_changed.assert_called_with("items", vm2.items)

def test_nested_recursive_sync():
    """Test that changes to nested BindableBase objects propagate."""
    class Child(BindableBase):
        title = BindableProperty(default="")
        
    class ParentVM(BaseViewModel):
        child = BindableProperty(default=None, sync_channel="child_sync")

    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = ParentVM(locator)
    vm2 = ParentVM(locator)
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    # Set child
    child_inst = Child()
    vm1.child = child_inst
    assert vm2.child is child_inst
    
    # Change nested property in vm1's child
    vm1.child.title = "Hello"
    
    # vm2 should see it
    assert vm2.child.title == "Hello"
    
    # Change it from vm2
    vm2.child.title = "World"
    assert vm1.child.title == "World"

def test_resubscription_on_replacement():
    """Test that sync continues after a property is replaced with a new instance."""
    class ReplaceVM(BaseViewModel):
        data = BindableProperty(default=None, sync_channel="data")

    locator = MockLocator()
    sync_mgr = ContextSyncManager(locator)
    locator.register_system(ContextSyncManager, sync_mgr)
    
    vm1 = ReplaceVM(locator)
    vm2 = ReplaceVM(locator)
    vm1.initialize_reactivity()
    vm2.initialize_reactivity()
    
    # 1. Set first instance
    list1 = BindableList([1])
    vm1.data = list1
    assert vm2.data is list1
    
    # 2. Replace with second instance in VM1
    list2 = BindableList([2])
    vm1.data = list2
    # vm2 should also receive the new instance via sync_channel
    assert vm2.data is list2
    
    # 3. Mutate second instance (Verify signals connected to new instance)
    vm1.data.append(3)
    assert vm1.data == [2, 3]
    assert vm2.data == [2, 3]
