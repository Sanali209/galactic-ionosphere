"""
Filter Editor Widgets - Dynamic UI for field-based filtering.

Creates appropriate editor widgets based on FieldDefinition type.
Used by FilterPanel for dynamic filter UI generation.
"""
from typing import Any, List, Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PySide6.QtCore import Signal, Qt
from loguru import logger

from uexplorer_src.viewmodels.field_registry import FieldDefinition, FieldType, Operator


class BaseFilterEditor(QWidget):
    """Base class for filter editors."""
    
    value_changed = Signal(str, str, object)  # field_name, operator, value
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(parent)
        self.field_def = field_def
        self._operator = field_def.get_operators()[0] if field_def.get_operators() else Operator.EQUALS
    
    def get_value(self) -> Any:
        """Get current filter value."""
        raise NotImplementedError
    
    def set_value(self, value: Any):
        """Set filter value."""
        raise NotImplementedError
    
    def get_operator(self) -> Operator:
        """Get current operator."""
        return self._operator
    
    def get_filter_dict(self) -> dict:
        """Get filter as MongoDB-compatible dict."""
        return {self.field_def.mongo_path: self.get_value()}
    
    def _emit_changed(self):
        """Emit value changed signal."""
        self.value_changed.emit(
            self.field_def.name,
            self._operator.value,
            self.get_value()
        )


class TextFilterEditor(BaseFilterEditor):
    """Text field editor with contains/equals/regex options."""
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Operator selector
        self.operator_combo = QComboBox()
        for op in self.field_def.get_operators():
            self.operator_combo.addItem(op.name.title(), op)
        self.operator_combo.currentIndexChanged.connect(self._on_operator_changed)
        layout.addWidget(self.operator_combo)
        
        # Text input
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(f"Filter {self.field_def.label}...")
        self.text_input.textChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.text_input, 1)
    
    def _on_operator_changed(self, index):
        self._operator = self.operator_combo.currentData()
        self._emit_changed()
    
    def get_value(self) -> str:
        return self.text_input.text()
    
    def set_value(self, value: str):
        self.text_input.setText(value or "")


class SelectFilterEditor(BaseFilterEditor):
    """Single/multi-select dropdown editor."""
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.combo = QComboBox()
        self.combo.addItem("-- Any --", None)
        
        if self.field_def.options:
            for value, label in self.field_def.options:
                self.combo.addItem(label, value)
        
        self.combo.currentIndexChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.combo)
    
    def get_value(self) -> Any:
        return self.combo.currentData()
    
    def set_value(self, value: Any):
        index = self.combo.findData(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)


class MultiSelectFilterEditor(BaseFilterEditor):
    """Multi-select checkbox list editor."""
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Operator selector
        op_row = QHBoxLayout()
        self.operator_combo = QComboBox()
        for op in self.field_def.get_operators():
            self.operator_combo.addItem(op.name.replace("_", " ").title(), op)
        self.operator_combo.currentIndexChanged.connect(self._on_operator_changed)
        op_row.addWidget(QLabel("Match:"))
        op_row.addWidget(self.operator_combo)
        op_row.addStretch()
        layout.addLayout(op_row)
        
        # Checkable list
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(100)
        
        if self.field_def.options:
            for value, label in self.field_def.options:
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, value)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.list_widget.addItem(item)
        
        self.list_widget.itemChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.list_widget)
    
    def _on_operator_changed(self, index):
        self._operator = self.operator_combo.currentData()
        self._emit_changed()
    
    def get_value(self) -> List[Any]:
        checked = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(item.data(Qt.ItemDataRole.UserRole))
        return checked
    
    def set_value(self, values: List[Any]):
        values = values or []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in values:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)


class RangeFilterEditor(BaseFilterEditor):
    """Numeric range editor with min/max inputs."""
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Operator
        self.operator_combo = QComboBox()
        for op in self.field_def.get_operators():
            self.operator_combo.addItem(op.name.replace("_", " ").title(), op)
        self.operator_combo.currentIndexChanged.connect(self._on_operator_changed)
        layout.addWidget(self.operator_combo)
        
        # Min value
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 999999)
        self.min_spin.setSpecialValueText("Min")
        self.min_spin.valueChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.min_spin)
        
        # Separator for BETWEEN
        self.sep_label = QLabel("-")
        layout.addWidget(self.sep_label)
        
        # Max value
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 999999)
        self.max_spin.setSpecialValueText("Max")
        self.max_spin.valueChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.max_spin)
        
        self._update_visibility()
    
    def _on_operator_changed(self, index):
        self._operator = self.operator_combo.currentData()
        self._update_visibility()
        self._emit_changed()
    
    def _update_visibility(self):
        """Show/hide max based on operator."""
        is_between = self._operator == Operator.BETWEEN
        self.sep_label.setVisible(is_between)
        self.max_spin.setVisible(is_between)
    
    def get_value(self):
        if self._operator == Operator.BETWEEN:
            return {"min": self.min_spin.value(), "max": self.max_spin.value()}
        return self.min_spin.value()
    
    def set_value(self, value):
        if isinstance(value, dict):
            self.min_spin.setValue(value.get("min", 0))
            self.max_spin.setValue(value.get("max", 0))
        else:
            self.min_spin.setValue(value or 0)


class BooleanFilterEditor(BaseFilterEditor):
    """Boolean toggle checkbox."""
    
    def __init__(self, field_def: FieldDefinition, parent=None):
        super().__init__(field_def, parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox(self.field_def.label)
        self.checkbox.setTristate(True)  # Unchecked = any, Checked = true, Partial = false
        self.checkbox.stateChanged.connect(lambda: self._emit_changed())
        layout.addWidget(self.checkbox)
    
    def get_value(self) -> Optional[bool]:
        state = self.checkbox.checkState()
        if state == Qt.CheckState.Checked:
            return True
        elif state == Qt.CheckState.PartiallyChecked:
            return False
        return None  # Any
    
    def set_value(self, value: Optional[bool]):
        if value is True:
            self.checkbox.setCheckState(Qt.CheckState.Checked)
        elif value is False:
            self.checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.checkbox.setCheckState(Qt.CheckState.Unchecked)


# Factory function to create editor by field type
EDITOR_MAP = {
    FieldType.TEXT: TextFilterEditor,
    FieldType.SELECT: SelectFilterEditor,
    FieldType.MULTI_SELECT: MultiSelectFilterEditor,
    FieldType.RANGE: RangeFilterEditor,
    FieldType.BOOLEAN: BooleanFilterEditor,
    # DATE and TAGS can be added later
}


def create_filter_editor(field_def: FieldDefinition, parent=None) -> BaseFilterEditor:
    """Create appropriate editor widget for field definition."""
    # Use custom factory if provided
    if field_def.editor_factory:
        return field_def.editor_factory(field_def, parent)
    
    # Use default editor by type
    editor_class = EDITOR_MAP.get(field_def.field_type, TextFilterEditor)
    return editor_class(field_def, parent)
