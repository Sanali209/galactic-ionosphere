"""
UExplorer - Rule Manager Dialog

Dialog for managing rules in the RulesEngine.
Supports CRUD operations, enable/disable, and priority ordering.
"""
from typing import Optional, List
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTextEdit,
    QCheckBox, QSpinBox, QGroupBox, QMessageBox, QSplitter,
    QWidget, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from loguru import logger


class RuleEditorWidget(QWidget):
    """Widget for editing a single rule."""
    
    rule_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_rule = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup editor UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Rule name
        name_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter rule name...")
        name_layout.addRow("Name:", self.name_input)
        layout.addLayout(name_layout)
        
        # Pattern group
        pattern_group = QGroupBox("Match Pattern")
        pattern_layout = QVBoxLayout(pattern_group)
        
        self.pattern_input = QLineEdit()
        self.pattern_input.setPlaceholderText("e.g., *.jpg, *screenshot*, etc.")
        pattern_layout.addWidget(self.pattern_input)
        
        self.is_regex = QCheckBox("Use Regular Expression")
        pattern_layout.addWidget(self.is_regex)
        
        layout.addWidget(pattern_group)
        
        # Action group
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        # Tags action
        tags_layout = QHBoxLayout()
        tags_layout.addWidget(QLabel("Add Tags:"))
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("tag1, tag2, tag3")
        tags_layout.addWidget(self.tags_input, 1)
        action_layout.addLayout(tags_layout)
        
        # Label action
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Set Label:"))
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("e.g., red, green, blue")
        label_layout.addWidget(self.label_input, 1)
        action_layout.addLayout(label_layout)
        
        # Rating action
        rating_layout = QHBoxLayout()
        rating_layout.addWidget(QLabel("Set Rating:"))
        self.rating_spin = QSpinBox()
        self.rating_spin.setRange(0, 5)
        self.rating_spin.setSpecialValueText("Don't set")
        rating_layout.addWidget(self.rating_spin)
        rating_layout.addStretch()
        action_layout.addLayout(rating_layout)
        
        layout.addWidget(action_group)
        
        # Priority
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("Priority:"))
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 1000)
        self.priority_spin.setValue(100)
        priority_layout.addWidget(self.priority_spin)
        
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        priority_layout.addStretch()
        priority_layout.addWidget(self.enabled_check)
        layout.addLayout(priority_layout)
        
        # Description
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Optional description...")
        self.desc_input.setMaximumHeight(80)
        desc_layout.addWidget(self.desc_input)
        layout.addWidget(desc_group)
        
        layout.addStretch()
    
    def load_rule(self, rule_data: dict):
        """Load rule data into editor."""
        self._current_rule = rule_data
        
        self.name_input.setText(rule_data.get("name", ""))
        self.pattern_input.setText(rule_data.get("pattern", ""))
        self.is_regex.setChecked(rule_data.get("is_regex", False))
        self.tags_input.setText(", ".join(rule_data.get("tags", [])))
        self.label_input.setText(rule_data.get("label", ""))
        self.rating_spin.setValue(rule_data.get("rating", 0))
        self.priority_spin.setValue(rule_data.get("priority", 100))
        self.enabled_check.setChecked(rule_data.get("enabled", True))
        self.desc_input.setPlainText(rule_data.get("description", ""))
    
    def get_rule_data(self) -> dict:
        """Get current rule data from editor."""
        tags = [t.strip() for t in self.tags_input.text().split(",") if t.strip()]
        
        return {
            "name": self.name_input.text().strip(),
            "pattern": self.pattern_input.text().strip(),
            "is_regex": self.is_regex.isChecked(),
            "tags": tags,
            "label": self.label_input.text().strip(),
            "rating": self.rating_spin.value() if self.rating_spin.value() > 0 else None,
            "priority": self.priority_spin.value(),
            "enabled": self.enabled_check.isChecked(),
            "description": self.desc_input.toPlainText().strip(),
        }
    
    def clear(self):
        """Clear all inputs."""
        self._current_rule = None
        self.name_input.clear()
        self.pattern_input.clear()
        self.is_regex.setChecked(False)
        self.tags_input.clear()
        self.label_input.clear()
        self.rating_spin.setValue(0)
        self.priority_spin.setValue(100)
        self.enabled_check.setChecked(True)
        self.desc_input.clear()


class RuleManagerDialog(QDialog):
    """
    Dialog for managing rules.
    
    Features:
    - List of all rules
    - Add/Edit/Delete rules
    - Enable/disable rules
    - Priority ordering
    """
    
    def __init__(self, locator=None, parent=None):
        super().__init__(parent)
        self._locator = locator
        self._rules_engine = None
        self._rules = []
        
        self.setWindowTitle("Rule Manager")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        self._load_rules_engine()
        self.refresh_rules()
        
        logger.info("RuleManagerDialog opened")
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Splitter for list and editor
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Rule list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        
        list_label = QLabel("Rules")
        list_label.setFont(QFont("", 12, QFont.Bold))
        list_layout.addWidget(list_label)
        
        self.rule_list = QListWidget()
        self.rule_list.currentItemChanged.connect(self._on_rule_selected)
        list_layout.addWidget(self.rule_list)
        
        # List buttons
        list_btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._on_add_rule)
        list_btn_layout.addWidget(self.btn_add)
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete_rule)
        list_btn_layout.addWidget(self.btn_delete)
        
        list_layout.addLayout(list_btn_layout)
        splitter.addWidget(list_widget)
        
        # Right: Rule editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        
        editor_label = QLabel("Edit Rule")
        editor_label.setFont(QFont("", 12, QFont.Bold))
        editor_layout.addWidget(editor_label)
        
        self.rule_editor = RuleEditorWidget()
        editor_layout.addWidget(self.rule_editor)
        
        # Save button
        self.btn_save = QPushButton("Save Rule")
        self.btn_save.clicked.connect(self._on_save_rule)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #5a7aaa;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
        """)
        editor_layout.addWidget(self.btn_save)
        
        splitter.addWidget(editor_widget)
        splitter.setSizes([300, 500])
        
        layout.addWidget(splitter)
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)
        
        # Apply dark theme
        self._apply_style()
    
    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QListWidget::item:selected {
                background-color: #5a7aaa;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QLineEdit, QTextEdit, QSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 12px;
            }
            QCheckBox {
                color: #cccccc;
            }
        """)
    
    def _load_rules_engine(self):
        """Get RulesEngine from locator."""
        if self._locator:
            try:
                from src.ucorefs.rules.engine import RulesEngine
                self._rules_engine = self._locator.get_system(RulesEngine)
            except (KeyError, ImportError) as e:
                logger.warning(f"RulesEngine not available: {e}")
    
    def refresh_rules(self):
        """Refresh rule list."""
        self.rule_list.clear()
        self._rules = []
        
        if self._rules_engine and hasattr(self._rules_engine, 'get_all_rules'):
            try:
                # This would be an async call in real implementation
                # For now, just show placeholder
                pass
            except Exception as e:
                logger.error(f"Failed to load rules: {e}")
        
        # Add placeholder rules for demo
        demo_rules = [
            {"name": "Tag Screenshots", "pattern": "*screenshot*", "enabled": True, "priority": 100},
            {"name": "Mark Raw Files", "pattern": "*.raw, *.cr2, *.nef", "enabled": True, "priority": 90},
        ]
        
        for rule in demo_rules:
            self._add_rule_to_list(rule)
    
    def _add_rule_to_list(self, rule: dict):
        """Add rule to list widget."""
        item = QListWidgetItem()
        
        name = rule.get("name", "Unnamed Rule")
        enabled = rule.get("enabled", True)
        
        icon = "✓" if enabled else "✗"
        item.setText(f"{icon} {name}")
        item.setData(Qt.UserRole, rule)
        
        self.rule_list.addItem(item)
        self._rules.append(rule)
    
    def _on_rule_selected(self, current, previous):
        """Handle rule selection."""
        if current:
            rule = current.data(Qt.UserRole)
            self.rule_editor.load_rule(rule)
    
    def _on_add_rule(self):
        """Add new rule."""
        self.rule_editor.clear()
        self.rule_editor.name_input.setFocus()
    
    def _on_delete_rule(self):
        """Delete selected rule."""
        current = self.rule_list.currentItem()
        if not current:
            return
        
        reply = QMessageBox.question(
            self, "Delete Rule",
            f"Delete rule '{current.text()}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            row = self.rule_list.row(current)
            self.rule_list.takeItem(row)
            if row < len(self._rules):
                del self._rules[row]
            logger.info(f"Rule deleted")
    
    def _on_save_rule(self):
        """Save current rule."""
        rule_data = self.rule_editor.get_rule_data()
        
        if not rule_data["name"]:
            QMessageBox.warning(self, "Validation Error", "Rule name is required.")
            return
        
        if not rule_data["pattern"]:
            QMessageBox.warning(self, "Validation Error", "Match pattern is required.")
            return
        
        # Update or add
        current = self.rule_list.currentItem()
        if current:
            current.setData(Qt.UserRole, rule_data)
            icon = "✓" if rule_data["enabled"] else "✗"
            current.setText(f"{icon} {rule_data['name']}")
        else:
            self._add_rule_to_list(rule_data)
        
        logger.info(f"Rule saved: {rule_data['name']}")
        QMessageBox.information(self, "Saved", "Rule saved successfully.")
