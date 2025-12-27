"""
UExplorer - Filter Tree Widget

Tree widget for building complex filter expressions using Q.AND/OR/NOT.
Users can visually construct filter conditions with drag-drop.
"""
from typing import Optional, List, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QComboBox, QLineEdit, QPushButton, QLabel, QSpinBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from loguru import logger


class FilterCondition:
    """Represents a single filter condition."""
    
    OPERATORS = {
        "equals": "=",
        "not_equals": "≠",
        "contains": "contains",
        "starts_with": "starts with",
        "greater_than": ">",
        "greater_eq": "≥",
        "less_than": "<",
        "less_eq": "≤",
        "in": "in",
    }
    
    FIELDS = [
        ("file_type", "File Type", ["image", "video", "audio", "document"]),
        ("rating", "Rating", None),  # None = use spinbox
        ("extension", "Extension", None),  # None = use text
        ("name", "Name", None),
    ]
    
    def __init__(self, field: str = "file_type", operator: str = "equals", value: Any = ""):
        self.field = field
        self.operator = operator
        self.value = value
    
    def to_q(self):
        """Convert to Q expression."""
        from src.ucorefs.query.builder import Q
        
        if self.field == "file_type":
            return Q.file_type(self.value)
        elif self.field == "rating":
            if self.operator == "greater_eq":
                return Q.rating_gte(int(self.value))
            elif self.operator == "less_eq":
                return Q.rating_lte(int(self.value))
        elif self.field == "name":
            return Q.name_contains(self.value)
        elif self.field == "extension":
            return Q.extension_in([self.value])
        
        # Default: field equals value
        return Q({self.field: self.value})
    
    def __str__(self):
        op_symbol = self.OPERATORS.get(self.operator, self.operator)
        return f"{self.field} {op_symbol} {self.value}"


from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox


class ConditionEditorDialog(QDialog):
    """Dialog for editing a filter condition."""
    
    def __init__(self, condition: FilterCondition = None, parent=None):
        super().__init__(parent)
        self.condition = condition or FilterCondition()
        self.setWindowTitle("Edit Condition")
        self.setMinimumWidth(300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout(self)
        
        # Field selector
        self.field_combo = QComboBox()
        for field_id, field_name, _ in FilterCondition.FIELDS:
            self.field_combo.addItem(field_name, field_id)
        idx = self.field_combo.findData(self.condition.field)
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        layout.addRow("Field:", self.field_combo)
        
        # Operator selector
        self.op_combo = QComboBox()
        for op_id, op_symbol in FilterCondition.OPERATORS.items():
            self.op_combo.addItem(f"{op_symbol} ({op_id})", op_id)
        idx = self.op_combo.findData(self.condition.operator)
        if idx >= 0:
            self.op_combo.setCurrentIndex(idx)
        layout.addRow("Operator:", self.op_combo)
        
        # Value input (combo for predefined, text for free-form)
        self.value_combo = QComboBox()
        self.value_combo.setEditable(True)
        self.value_combo.setEditText(str(self.condition.value))
        layout.addRow("Value:", self.value_combo)
        
        self._update_value_options()
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self._apply_style()
    
    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QLabel { color: #ffffff; }
            QComboBox { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555; padding: 4px; }
            QDialogButtonBox QPushButton { background-color: #5a7aaa; color: white; padding: 6px 12px; }
        """)
    
    def _on_field_changed(self, index):
        self._update_value_options()
    
    def _update_value_options(self):
        field_id = self.field_combo.currentData()
        self.value_combo.clear()
        
        # Find field definition
        for fid, fname, options in FilterCondition.FIELDS:
            if fid == field_id and options:
                self.value_combo.addItems(options)
                break
    
    def get_condition(self) -> FilterCondition:
        """Get the configured condition."""
        return FilterCondition(
            field=self.field_combo.currentData(),
            operator=self.op_combo.currentData(),
            value=self.value_combo.currentText()
        )


class FilterTreeWidget(QWidget):
    """
    Tree widget for building filter expressions (Q.AND/OR/NOT).
    
    Signals:
        query_changed: Emitted when the filter tree changes
    """
    
    query_changed = Signal(object)  # Q expression
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        logger.info("FilterTreeWidget initialized")
    
    def setup_ui(self):
        """Build UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Filter Expression"])
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add_and = QPushButton("+ AND")
        self.btn_add_and.clicked.connect(lambda: self._add_group("AND"))
        btn_layout.addWidget(self.btn_add_and)
        
        self.btn_add_or = QPushButton("+ OR")
        self.btn_add_or.clicked.connect(lambda: self._add_group("OR"))
        btn_layout.addWidget(self.btn_add_or)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        btn_layout.addWidget(self.btn_clear)
        
        layout.addLayout(btn_layout)
        
        self._apply_style()
    
    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #4a4a4a;
            }
            QTreeWidget::item:selected {
                background-color: #5a7aaa;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
    
    def _add_group(self, group_type: str, parent: QTreeWidgetItem = None):
        """Add AND/OR/NOT group."""
        item = QTreeWidgetItem()
        item.setText(0, group_type)
        item.setData(0, Qt.UserRole, {"type": group_type, "children": []})
        item.setExpanded(True)
        
        if parent:
            parent.addChild(item)
        else:
            self.tree.addTopLevelItem(item)
        
        self._emit_query_changed()
        return item
    
    def _add_condition(self, parent: QTreeWidgetItem):
        """Add condition to group with dialog."""
        dialog = ConditionEditorDialog(parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        condition = dialog.get_condition()
        item = QTreeWidgetItem()
        item.setText(0, str(condition))
        item.setData(0, Qt.UserRole, {"type": "condition", "condition": condition})
        
        if parent:
            parent.addChild(item)
        else:
            # Add to first group or create one
            if self.tree.topLevelItemCount() == 0:
                group = self._add_group("AND")
                group.addChild(item)
            else:
                self.tree.topLevelItem(0).addChild(item)
        
        self._emit_query_changed()
    
    def _edit_condition(self, item: QTreeWidgetItem):
        """Edit existing condition."""
        data = item.data(0, Qt.UserRole)
        if not data or data.get("type") != "condition":
            return
        
        condition = data.get("condition")
        dialog = ConditionEditorDialog(condition=condition, parent=self)
        if dialog.exec() == QDialog.Accepted:
            new_condition = dialog.get_condition()
            item.setText(0, str(new_condition))
            item.setData(0, Qt.UserRole, {"type": "condition", "condition": new_condition})
            self._emit_query_changed()
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click to edit conditions."""
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "condition":
            self._edit_condition(item)
    
    def _show_context_menu(self, position):
        """Show right-click menu."""
        item = self.tree.itemAt(position)
        menu = QMenu(self)
        
        if item:
            data = item.data(0, Qt.UserRole)
            if data and data.get("type") in ("AND", "OR", "NOT"):
                menu.addAction("Add Condition", lambda: self._add_condition(item))
                menu.addAction("Add AND Group", lambda: self._add_group("AND", item))
                menu.addAction("Add OR Group", lambda: self._add_group("OR", item))
                menu.addAction("Add NOT Group", lambda: self._add_group("NOT", item))
                menu.addSeparator()
            elif data and data.get("type") == "condition":
                menu.addAction("Edit Condition", lambda: self._edit_condition(item))
                menu.addSeparator()
            
            menu.addAction("Delete", lambda: self._delete_item(item))
        else:
            menu.addAction("Add AND Group", lambda: self._add_group("AND"))
            menu.addAction("Add OR Group", lambda: self._add_group("OR"))
        
        menu.exec(self.tree.mapToGlobal(position))
    
    def _delete_item(self, item: QTreeWidgetItem):
        """Delete item from tree."""
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            idx = self.tree.indexOfTopLevelItem(item)
            self.tree.takeTopLevelItem(idx)
        
        self._emit_query_changed()
    
    def clear(self):
        """Clear all filters."""
        self.tree.clear()
        self._emit_query_changed()
    
    def get_query(self):
        """
        Build Q expression from tree.
        
        Returns:
            Q expression or None if empty
        """
        from src.ucorefs.query.builder import Q
        
        if self.tree.topLevelItemCount() == 0:
            return None
        
        return self._build_query_from_item(self.tree.topLevelItem(0))
    
    def _build_query_from_item(self, item: QTreeWidgetItem):
        """Recursively build Q expression."""
        from src.ucorefs.query.builder import Q
        
        data = item.data(0, Qt.UserRole)
        if not data:
            return None
        
        item_type = data.get("type")
        
        if item_type == "condition":
            condition = data.get("condition")
            return condition.to_q() if condition else None
        
        # Group: AND/OR/NOT
        children_queries = []
        for i in range(item.childCount()):
            child_q = self._build_query_from_item(item.child(i))
            if child_q:
                children_queries.append(child_q)
        
        if not children_queries:
            return None
        
        if item_type == "AND":
            return Q.AND(*children_queries)
        elif item_type == "OR":
            return Q.OR(*children_queries)
        elif item_type == "NOT":
            return Q.NOT(children_queries[0]) if children_queries else None
        
        return None
    
    def _emit_query_changed(self):
        """Emit query changed signal."""
        query = self.get_query()
        self.query_changed.emit(query)
