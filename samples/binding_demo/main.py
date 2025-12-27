"""
Binding Demo Sample Application.

Demonstrates WPF-style data binding in a Foundation application.
Shows:
- BindableProperty for automatic change notification
- bind() for declarative property synchronization
- DataContextMixin for ViewModel inheritance

Run with: python main.py
"""
import sys
import asyncio
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QGroupBox
)
from PySide6.QtCore import Qt, Signal

# Foundation imports
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/samples/", 1)[0])
from src.ui.mvvm.bindable import BindableBase, BindableProperty
from src.ui.mvvm.binding import bind, bind_command, BindingMode
from src.ui.mvvm.data_context import DataContextMixin

# =============================================================================
# ViewModel
# =============================================================================

class DemoViewModel(BindableBase):
    """
    ViewModel demonstrating BindableProperty usage.
    
    Properties emit signals automatically when changed.
    """
    
    # Specific signals for properties (optional but recommended for type safety)
    nameChanged = Signal(str)
    ageChanged = Signal(int)
    greetingChanged = Signal(str)
    
    # BindableProperty - auto-emits signals on change
    name = BindableProperty(default="World")
    age = BindableProperty(default=25, coerce=lambda x: max(0, min(120, int(x))))
    
    # Computed property (manual signal emission)
    @property
    def greeting(self) -> str:
        return f"Hello, {self.name}! You are {self.age} years old."
    
    def __init__(self):
        super().__init__()
        # Connect to update greeting when name or age changes
        self.propertyChanged.connect(self._on_property_changed)
    
    def _on_property_changed(self, prop_name: str, value):
        """Update greeting when dependencies change."""
        if prop_name in ("name", "age"):
            self.greetingChanged.emit(self.greeting)
    
    def reset(self):
        """Reset all values to defaults."""
        self.name = "World"
        self.age = 25

# =============================================================================
# View
# =============================================================================

class DemoWindow(DataContextMixin, QMainWindow):
    """
    Main window demonstrating data binding patterns.
    """
    
    def __init__(self, viewmodel: DemoViewModel):
        super().__init__()
        self.setWindowTitle("Data Binding Demo")
        self.resize(500, 400)
        
        # Set DataContext (available to all children)
        self.set_data_context(viewmodel)
        self.vm = viewmodel
        
        self._setup_ui()
        self._setup_bindings()
    
    def _setup_ui(self):
        """Build the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Input Group
        input_group = QGroupBox("Input (Two-Way Binding)")
        input_layout = QVBoxLayout(input_group)
        
        # Name input
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        input_layout.addLayout(name_row)
        
        # Age input (slider + spinbox)
        age_row = QHBoxLayout()
        age_row.addWidget(QLabel("Age:"))
        self.age_slider = QSlider(Qt.Horizontal)
        self.age_slider.setRange(0, 120)
        age_row.addWidget(self.age_slider)
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setRange(0, 120)
        age_row.addWidget(self.age_spinbox)
        input_layout.addLayout(age_row)
        
        layout.addWidget(input_group)
        
        # Output Group
        output_group = QGroupBox("Output (One-Way Binding)")
        output_layout = QVBoxLayout(output_group)
        
        self.greeting_label = QLabel()
        self.greeting_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        output_layout.addWidget(self.greeting_label)
        
        # Debug labels
        self.name_debug = QLabel()
        self.age_debug = QLabel()
        output_layout.addWidget(self.name_debug)
        output_layout.addWidget(self.age_debug)
        
        layout.addWidget(output_group)
        
        # Actions
        actions_group = QGroupBox("Actions (Command Binding)")
        actions_layout = QHBoxLayout(actions_group)
        
        self.reset_button = QPushButton("Reset Values")
        actions_layout.addWidget(self.reset_button)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
    
    def _setup_bindings(self):
        """Set up data bindings between VM and View."""
        vm = self.vm
        
        # Two-way bindings (VM <-> Widget)
        bind(vm, "name", self.name_edit, "text", mode=BindingMode.TWO_WAY)
        bind(vm, "age", self.age_slider, "value", mode=BindingMode.TWO_WAY)
        bind(vm, "age", self.age_spinbox, "value", mode=BindingMode.TWO_WAY)
        
        # One-way bindings (VM -> Widget) with converters
        bind(
            vm, "name", self.name_debug, "text",
            converter=lambda x: f"Name value: '{x}'"
        )
        bind(
            vm, "age", self.age_debug, "text",
            converter=lambda x: f"Age value: {x}"
        )
        
        # Manual connection for computed property
        vm.greetingChanged.connect(self.greeting_label.setText)
        self.greeting_label.setText(vm.greeting)
        
        # Command binding
        bind_command(vm, "reset", self.reset_button, "clicked")

# =============================================================================
# Main
# =============================================================================

def main():
    app = QApplication(sys.argv)
    
    # Create ViewModel
    vm = DemoViewModel()
    
    # Create View with ViewModel
    window = DemoWindow(vm)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
