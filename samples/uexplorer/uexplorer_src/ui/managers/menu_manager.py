"""
MenuManager - Centralized Menu Construction

Extracts menu building logic from MainWindow to follow Single Responsibility Principle.
"""
from PySide6.QtWidgets import QMainWindow, QMenuBar
from loguru import logger


class MenuManager:
    """
    Manages menu bar construction and actions.
    
    This class encapsulates all menu-related logic that was previously
    embedded in MainWindow, improving maintainability and testability.
    """
    
    def __init__(self, main_window: QMainWindow, action_registry):
        """
        Initialize MenuManager.
        
        Args:
            main_window: The main window to attach menus to
            action_registry: ActionRegistry containing all available actions
        """
        self.window = main_window
        self.action_registry = action_registry
    
    def build_menus(self) -> QMenuBar:
        """
        Build and attach the complete menu bar.
        
        Returns:
            The constructed QMenuBar
        """
        menubar = self.window.menuBar()
        
        self._build_file_menu(menubar)
        self._build_edit_menu(menubar)
        self._build_view_menu(menubar)
        self._build_tools_menu(menubar)
        self._build_maintenance_menu(menubar)
        self._build_help_menu(menubar)
        
        logger.info("MenuManager: Menu bar constructed")
        return menubar
    
    def _build_file_menu(self, menubar: QMenuBar):
        """Build File menu."""
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_registry.get_action("file.new_window"))
        file_menu.addAction(self.action_registry.get_action("file.new_browser"))
        file_menu.addSeparator()
        file_menu.addAction(self.action_registry.get_action("file.exit"))
    
    def _build_edit_menu(self, menubar: QMenuBar):
        """Build Edit menu."""
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_registry.get_action("edit.settings"))
    
    def _build_view_menu(self, menubar: QMenuBar):
        """Build View menu with panels submenu."""
        view_menu = menubar.addMenu("&View")
        
        # Dashboard
        view_menu.addAction(self.action_registry.get_action("view.dashboard"))
        
        # Panels submenu
        panels_menu = view_menu.addMenu("&Panels")
        panels_menu.addAction(self.action_registry.get_action("view.panel.tags"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.albums"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.directories"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.relations"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.similar"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.annotation"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.properties"))
        panels_menu.addSeparator()
        panels_menu.addAction(self.action_registry.get_action("view.panel.search"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.background"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.maintenance"))
        
        view_menu.addSeparator()
        
        # Split actions
        view_menu.addAction(self.action_registry.get_action("view.split_horizontal"))
        view_menu.addAction(self.action_registry.get_action("view.split_vertical"))
        view_menu.addAction(self.action_registry.get_action("view.close_split"))
        
        view_menu.addSeparator()
        view_menu.addAction(self.action_registry.get_action("view.reset_layout"))
        
        view_menu.addSeparator()
        view_menu.addAction(self.action_registry.get_action("view.thumbnails"))
    
    def _build_tools_menu(self, menubar: QMenuBar):
        """Build Tools menu."""
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.action_registry.get_action("tools.scan"))
        tools_menu.addAction(self.action_registry.get_action("tools.reprocess"))
        tools_menu.addAction(self.action_registry.get_action("tools.reindex_all"))
        tools_menu.addSeparator()
        tools_menu.addAction(self.action_registry.get_action("tools.library"))
        tools_menu.addAction(self.action_registry.get_action("tools.rules"))
        tools_menu.addSeparator()
        tools_menu.addAction(self.action_registry.get_action("tools.command_palette"))
    
    def _build_maintenance_menu(self, menubar: QMenuBar):
        """Build Maintenance menu for data integrity operations."""
        maintenance_menu = menubar.addMenu("&Maintenance")
        maintenance_menu.addAction(self.action_registry.get_action("maintenance.rebuild_counts"))
        maintenance_menu.addSeparator()
        maintenance_menu.addAction(self.action_registry.get_action("maintenance.verify_references"))
        maintenance_menu.addAction(self.action_registry.get_action("maintenance.cleanup_orphaned"))
    
    def _build_help_menu(self, menubar: QMenuBar):
        """Build Help menu."""
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.action_registry.get_action("help.shortcuts"))
        help_menu.addSeparator()
        help_menu.addAction(self.action_registry.get_action("help.about"))
