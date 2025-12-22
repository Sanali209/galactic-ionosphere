"""
Unit Tests for WPF-Style Data Binding System.

Tests for:
- BindableProperty descriptor
- bind() function
- DataContextMixin
"""
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget, QLineEdit, QLabel

from src.ui.mvvm.bindable import BindableProperty, BindableBase
from src.ui.mvvm.binding import bind, bind_command, BindingMode
from src.ui.mvvm.data_context import DataContextMixin, BindableWidget


# Ensure QApplication exists for Qt tests
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# =============================================================================
# BindableProperty Tests
# =============================================================================

class TestBindableProperty:
    """Tests for BindableProperty descriptor."""
    
    def test_default_value(self, qapp):
        """Test that default value is returned when not set."""
        class TestVM(BindableBase):
            name = BindableProperty(default="default_name")
        
        vm = TestVM()
        assert vm.name == "default_name"
    
    def test_set_value(self, qapp):
        """Test setting a value."""
        class TestVM(BindableBase):
            name = BindableProperty(default="")
        
        vm = TestVM()
        vm.name = "Alice"
        assert vm.name == "Alice"
    
    def test_emits_property_changed(self, qapp):
        """Test that generic propertyChanged signal is emitted."""
        class TestVM(BindableBase):
            count = BindableProperty(default=0)
        
        vm = TestVM()
        callback = MagicMock()
        vm.propertyChanged.connect(callback)
        
        vm.count = 42
        
        callback.assert_called_once_with("count", 42)
    
    def test_emits_specific_signal(self, qapp):
        """Test that specific signal is emitted if defined."""
        class TestVM(BindableBase):
            countChanged = Signal(int)
            count = BindableProperty(default=0)
        
        vm = TestVM()
        callback = MagicMock()
        vm.countChanged.connect(callback)
        
        vm.count = 100
        
        callback.assert_called_once_with(100)
    
    def test_no_emit_on_same_value(self, qapp):
        """Test that signal is NOT emitted when value doesn't change."""
        class TestVM(BindableBase):
            name = BindableProperty(default="same")
        
        vm = TestVM()
        callback = MagicMock()
        vm.propertyChanged.connect(callback)
        
        vm.name = "same"  # Same as default
        
        callback.assert_not_called()
    
    def test_coerce_function(self, qapp):
        """Test value coercion."""
        class TestVM(BindableBase):
            age = BindableProperty(default=0, coerce=lambda x: max(0, int(x)))
        
        vm = TestVM()
        vm.age = -5
        assert vm.age == 0
        
        vm.age = "25"
        assert vm.age == 25


# =============================================================================
# bind() Function Tests
# =============================================================================

class TestBindFunction:
    """Tests for bind() helper function."""
    
    def test_one_way_binding(self, qapp):
        """Test one-way binding from VM to widget."""
        class TestVM(BindableBase):
            textChanged = Signal(str)
            text = BindableProperty(default="")
        
        vm = TestVM()
        label = QLabel()
        
        bind(vm, "text", label, "text", mode=BindingMode.ONE_WAY)
        
        vm.text = "Hello"
        assert label.text() == "Hello"
    
    def test_initial_sync(self, qapp):
        """Test that initial value is synced on bind."""
        class TestVM(BindableBase):
            textChanged = Signal(str)
            text = BindableProperty(default="Initial")
        
        vm = TestVM()
        label = QLabel()
        
        bind(vm, "text", label, "text")
        
        assert label.text() == "Initial"
    
    def test_two_way_binding(self, qapp):
        """Test two-way binding between VM and widget."""
        class TestVM(BindableBase):
            textChanged = Signal(str)
            text = BindableProperty(default="")
        
        vm = TestVM()
        line_edit = QLineEdit()
        
        bind(vm, "text", line_edit, "text", mode=BindingMode.TWO_WAY)
        
        # VM -> Widget
        vm.text = "From VM"
        assert line_edit.text() == "From VM"
        
        # Widget -> VM
        line_edit.setText("From Widget")
        # Note: We need to emit the signal manually in tests
        line_edit.textChanged.emit("From Widget")
        assert vm.text == "From Widget"
    
    def test_one_time_binding(self, qapp):
        """Test one-time binding only syncs initial value."""
        class TestVM(BindableBase):
            textChanged = Signal(str)
            text = BindableProperty(default="Initial")
        
        vm = TestVM()
        label = QLabel()
        
        bind(vm, "text", label, "text", mode=BindingMode.ONE_TIME)
        
        assert label.text() == "Initial"
        
        vm.text = "Updated"
        # Should NOT update because it's one-time
        assert label.text() == "Initial"
    
    def test_converter(self, qapp):
        """Test value converter in binding."""
        class TestVM(BindableBase):
            countChanged = Signal(int)
            count = BindableProperty(default=0)
        
        vm = TestVM()
        label = QLabel()
        
        bind(
            vm, "count", label, "text", 
            converter=lambda x: f"Count: {x}"
        )
        
        vm.count = 42
        assert label.text() == "Count: 42"


class TestBindCommand:
    """Tests for bind_command() function."""
    
    def test_bind_command(self, qapp):
        """Test binding a command to a button click."""
        class TestVM(BindableBase):
            def save(self):
                self.save_called = True
        
        vm = TestVM()
        vm.save_called = False
        
        # Create a mock button with clicked signal
        class MockButton(QObject):
            clicked = Signal()
        
        button = MockButton()
        bind_command(vm, "save", button, "clicked")
        
        button.clicked.emit()
        
        assert vm.save_called is True


# =============================================================================
# DataContextMixin Tests
# =============================================================================

class TestDataContextMixin:
    """Tests for DataContext functionality."""
    
    def test_set_and_get_data_context(self, qapp):
        """Test setting and getting DataContext."""
        class TestWidget(DataContextMixin, QWidget):
            pass
        
        class TestVM(BindableBase):
            pass
        
        vm = TestVM()
        widget = TestWidget()
        widget.set_data_context(vm)
        
        assert widget.get_data_context() is vm
    
    def test_inherit_from_parent(self, qapp):
        """Test that child widgets inherit DataContext from parent."""
        class ParentWidget(DataContextMixin, QWidget):
            pass
        
        class ChildWidget(DataContextMixin, QWidget):
            pass
        
        class TestVM(BindableBase):
            pass
        
        vm = TestVM()
        parent = ParentWidget()
        child = ChildWidget(parent)
        
        parent.set_data_context(vm)
        
        # Child should get VM from parent
        assert child.get_data_context() is vm
    
    def test_typed_data_context(self, qapp):
        """Test getting typed DataContext."""
        class TestWidget(DataContextMixin, QWidget):
            pass
        
        class SpecificVM(BindableBase):
            pass
        
        class OtherVM(BindableBase):
            pass
        
        vm = SpecificVM()
        widget = TestWidget()
        widget.set_data_context(vm)
        
        # Correct type
        assert widget.get_typed_data_context(SpecificVM) is vm
        
        # Wrong type
        assert widget.get_typed_data_context(OtherVM) is None
