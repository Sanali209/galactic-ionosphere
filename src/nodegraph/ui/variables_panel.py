# -*- coding: utf-8 -*-
"""
VariablesPanel - Panel for managing graph variables.

Displays and allows editing of graph-level variables.
"""
from typing import Optional, Dict, TYPE_CHECKING
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QFrame,
    QMessageBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QAction

if TYPE_CHECKING:
    from ..core.graph import NodeGraph, Variable


class VariablesPanel(QWidget):
    """
    Panel for managing graph variables.
    
    Features:
    - List of current variables
    - Add/remove variables
    - Edit variable name, type, default value
    
    Signals:
        variable_added: Emitted when variable is added (name)
        variable_removed: Emitted when variable is removed (name)
        variable_changed: Emitted when variable is modified (name)
    """
    
    variable_added = Signal(str)
    variable_removed = Signal(str)
    variable_changed = Signal(str)
    
    # Available variable types
    VAR_TYPES = ["String", "Integer", "Float", "Boolean", "Array"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: Optional['NodeGraph'] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Variables")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Add button
        add_btn = QPushButton("+ Add Variable")
        add_btn.clicked.connect(self._add_variable)
        layout.addWidget(add_btn)
        
        # Variables list
        self._list = QListWidget()
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemDoubleClicked.connect(self._edit_variable)
        layout.addWidget(self._list)
        
        # Edit form (hidden initially)
        self._edit_frame = QFrame()
        self._edit_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        edit_layout = QFormLayout(self._edit_frame)
        
        self._name_edit = QLineEdit()
        edit_layout.addRow("Name:", self._name_edit)
        
        self._type_combo = QComboBox()
        self._type_combo.addItems(self.VAR_TYPES)
        edit_layout.addRow("Type:", self._type_combo)
        
        self._default_edit = QLineEdit()
        edit_layout.addRow("Default:", self._default_edit)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_edit)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel_edit)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        edit_layout.addRow(btn_layout)
        
        self._edit_frame.hide()
        layout.addWidget(self._edit_frame)
        
        self._editing_var: Optional[str] = None
    
    def set_graph(self, graph: Optional['NodeGraph']):
        """Set the graph to manage variables for."""
        self._graph = graph
        self._refresh_list()
    
    def _refresh_list(self):
        """Refresh the variables list from graph."""
        self._list.clear()
        
        if not self._graph:
            return
        
        for name, var in self._graph.variables.items():
            item = QListWidgetItem(f"{name} ({var.var_type}) = {var.default_value}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._list.addItem(item)
    
    def _add_variable(self):
        """Show form to add new variable."""
        self._editing_var = None
        self._name_edit.setText("")
        self._type_combo.setCurrentIndex(0)
        self._default_edit.setText("")
        self._edit_frame.show()
        self._name_edit.setFocus()
    
    def _edit_variable(self, item: QListWidgetItem):
        """Edit selected variable."""
        if not self._graph:
            return
        
        name = item.data(Qt.ItemDataRole.UserRole)
        var = self._graph.variables.get(name)
        if not var:
            return
        
        self._editing_var = name
        self._name_edit.setText(var.name)
        
        idx = self._type_combo.findText(var.var_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        
        self._default_edit.setText(str(var.default_value) if var.default_value is not None else "")
        self._edit_frame.show()
    
    def _save_edit(self):
        """Save variable edit."""
        if not self._graph:
            return
        
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Variable name is required")
            return
        
        var_type = self._type_combo.currentText()
        default_str = self._default_edit.text()
        
        # Parse default value based on type
        default_value = self._parse_default(var_type, default_str)
        
        # Remove old if renaming
        if self._editing_var and self._editing_var != name:
            self._graph.remove_variable(self._editing_var)
        
        # Add/update variable
        self._graph.add_variable(name, var_type, default_value)
        
        self._edit_frame.hide()
        self._refresh_list()
        
        if self._editing_var:
            self.variable_changed.emit(name)
        else:
            self.variable_added.emit(name)
    
    def _cancel_edit(self):
        """Cancel variable edit."""
        self._edit_frame.hide()
        self._editing_var = None
    
    def _parse_default(self, var_type: str, value_str: str):
        """Parse default value string to appropriate type."""
        if not value_str:
            return None
        
        try:
            if var_type == "Integer":
                return int(value_str)
            elif var_type == "Float":
                return float(value_str)
            elif var_type == "Boolean":
                return value_str.lower() in ("true", "1", "yes")
            elif var_type == "Array":
                return value_str.split(",") if value_str else []
            else:
                return value_str
        except ValueError:
            return value_str
    
    def _show_context_menu(self, pos):
        """Show context menu for variable list."""
        item = self._list.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        
        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(lambda: self._edit_variable(item))
        
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_variable(item))
        
        menu.exec(self._list.mapToGlobal(pos))
    
    def _delete_variable(self, item: QListWidgetItem):
        """Delete a variable."""
        if not self._graph:
            return
        
        name = item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, "Delete Variable",
            f"Delete variable '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._graph.remove_variable(name)
            self._refresh_list()
            self.variable_removed.emit(name)
