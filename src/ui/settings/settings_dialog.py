"""
Settings dialog with category tree and ConfigManager integration.
"""
from typing import Dict, Any, List
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, 
                                QTreeWidgetItem, QStackedWidget, QLineEdit, 
                                QPushButton, QLabel, QWidget, QSplitter)
from PySide6.QtCore import Qt, Signal
from loguru import logger

class SettingsDialog(QDialog):
    """
    VS Code-style settings dialog.
    Left side: Category tree
    Right side: Settings widgets for selected category
    """
    settings_changed = Signal(str, object)  # category, value
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("Settings")
        self.resize(900, 600)
        
        self._category_widgets: Dict[str, QWidget] = {}
        self._init_ui()
        
        logger.info("SettingsDialog initialized")
    
    def _init_ui(self):
        """Build the settings dialog UI."""
        layout = QVBoxLayout(self)
        
        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search settings...")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Main splitter (category tree | settings panel)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Category tree
        self.category_tree = QTreeWidget()
        self.category_tree.setHeaderLabel("Categories")
        self.category_tree.currentItemChanged.connect(self._on_category_selected)
        splitter.addWidget(self.category_tree)
        
        # Right: Stacked widget for settings panels
        self.settings_stack = QStackedWidget()
        splitter.addWidget(self.settings_stack)
        
        splitter.setSizes([250, 650])
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Populate categories
        self._build_categories()
    
    def _build_categories(self):
        """Build category tree from ConfigManager structure."""
        # General category
        general_item = QTreeWidgetItem(self.category_tree, ["General"])
        general_widget = self._create_general_settings()
        self._add_category("General", general_item, general_widget)
        
        # Editor category (placeholder)
        editor_item = QTreeWidgetItem(self.category_tree, ["Editor"])
        editor_widget = QLabel("Editor settings coming soon...")
        self._add_category("Editor", editor_item, editor_widget)
        
        # Appearance category (placeholder)
        appearance_item = QTreeWidgetItem(self.category_tree, ["Appearance"])
        appearance_widget = QLabel("Appearance settings coming soon...")
        self._add_category("Appearance", appearance_item, appearance_widget)
        
        # Select first category
        self.category_tree.setCurrentItem(general_item)
    
    def _add_category(self, name: str, tree_item: QTreeWidgetItem, widget: QWidget):
        """Add a category with its settings widget."""
        self._category_widgets[name] = widget
        self.settings_stack.addWidget(widget)
        tree_item.setData(0, Qt.UserRole, name)
    
    def _create_general_settings(self) -> QWidget:
        """Create General settings panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("<b>General Settings</b>"))
        layout.addSpacing(10)
        
        # App Name setting
        layout.addWidget(QLabel("Application Name:"))
        self.app_name_input = QLineEdit()
        try:
            self.app_name_input.setText(self.config.data.app_name)
        except:
            self.app_name_input.setText("Foundation App")
        self.app_name_input.textChanged.connect(
            lambda text: self._on_setting_changed("app_name", text))
        layout.addWidget(self.app_name_input)
        
        layout.addSpacing(20)
        
        # Database Name setting  
        layout.addWidget(QLabel("Database Name:"))
        self.db_name_input = QLineEdit()
        try:
            self.db_name_input.setText(self.config.data.db_name)
        except:
            self.db_name_input.setText("foundation_db")
        self.db_name_input.textChanged.connect(
            lambda text: self._on_setting_changed("db_name", text))
        layout.addWidget(self.db_name_input)
        
        layout.addStretch()
        return widget
    
    def _on_category_selected(self, current: QTreeWidgetItem, previous: QTreeWidgetItem):
        """Handle category selection."""
        if not current:
            return
        
        category_name = current.data(0, Qt.UserRole)
        if category_name in self._category_widgets:
            widget = self._category_widgets[category_name]
            self.settings_stack.setCurrentWidget(widget)
    
    def _on_search(self, text: str):
        """Filter settings by search text."""
        # Simple implementation: show/hide categories
        if not text:
            # Show all
            for i in range(self.category_tree.topLevelItemCount()):
                self.category_tree.topLevelItem(i).setHidden(False)
        else:
            # Filter
            text_lower = text.lower()
            for i in range(self.category_tree.topLevelItemCount()):
                item = self.category_tree.topLevelItem(i)
                category_name = item.text(0).lower()
                item.setHidden(text_lower not in category_name)
    
    def _on_setting_changed(self, key: str, value: Any):
        """Handle setting value change."""
        logger.debug(f"Setting changed: {key} = {value}")
        self.settings_changed.emit(key, value)
        
        # Update ConfigManager
        try:
            setattr(self.config.data, key, value)
        except Exception as e:
            logger.warning(f"Could not set config.{key}: {e}")
    
    def _on_reset(self):
        """Reset current category to defaults."""
        current = self.category_tree.currentItem()
        if not current:
            return
        
        category_name = current.data(0, Qt.UserRole)
        logger.info(f"Reset settings for category: {category_name}")
        
        # TODO: Implement reset to defaults from Pydantic model
        # For now, just log
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Reset", 
                               f"Reset {category_name} settings to defaults.\n(Feature coming soon)")
