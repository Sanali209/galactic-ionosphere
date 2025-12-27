"""
ToolbarManager - Centralized Toolbar Construction

Extracts toolbar building logic from MainWindow to follow Single Responsibility Principle.
"""
from PySide6.QtWidgets import QMainWindow, QToolBar
from PySide6.QtCore import QSize
from PySide6.QtGui import QAction
from loguru import logger


class ToolbarManager:
    """
    Manages toolbar construction and actions.
    
    This class encapsulates all toolbar-related logic that was previously
    embedded in MainWindow, improving maintainability and testability.
    """
    
    def __init__(self, main_window: QMainWindow, action_registry=None):
        """
        Initialize ToolbarManager.
        
        Args:
            main_window: The main window to attach toolbars to
            action_registry: Optional ActionRegistry for action-based toolbar items
        """
        self.window = main_window
        self.action_registry = action_registry
        self.toolbar = None
    
    def build_toolbar(self) -> QToolBar:
        """
        Build and attach the main toolbar.
        
        Returns:
            The constructed QToolBar
        """
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.window.addToolBar(self.toolbar)
        
        # Scan button
        scan_btn = QAction("Scan", self.window)
        scan_btn.setToolTip("Scan directories for changes")
        self.toolbar.addAction(scan_btn)
        
        self.toolbar.addSeparator()
        
        # Tags button
        tags_btn = QAction("Tags", self.window)
        tags_btn.setToolTip("Manage tags")
        self.toolbar.addAction(tags_btn)
        
        # Albums button
        albums_btn = QAction("Albums", self.window)
        albums_btn.setToolTip("Manage albums")
        self.toolbar.addAction(albums_btn)
        
        self.toolbar.addSeparator()
        
        # Rules button
        rules_btn = QAction("Rules", self.window)
        rules_btn.setToolTip("Manage automation rules")
        self.toolbar.addAction(rules_btn)
        
        # Search button
        search_btn = QAction("Search", self.window)
        search_btn.setToolTip("Advanced search")
        self.toolbar.addAction(search_btn)
        
        logger.info("ToolbarManager: Toolbar constructed")
        return self.toolbar
    
    def add_action(self, action_id: str):
        """
        Add an action from the registry to the toolbar.
        
        Args:
            action_id: ID of the action to add
        """
        if self.action_registry and self.toolbar:
            action = self.action_registry.get_action(action_id)
            if action:
                self.toolbar.addAction(action)
    
    def add_separator(self):
        """Add a separator to the toolbar."""
        if self.toolbar:
            self.toolbar.addSeparator()
