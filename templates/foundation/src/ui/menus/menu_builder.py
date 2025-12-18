"""
Menu builder for creating application menus.
"""
from PySide6.QtWidgets import QMainWindow, QMenu
from loguru import logger
from .action_registry import ActionRegistry

class MenuBuilder:
    """
    Builds application menu bar from action registry.
    """
    def __init__(self, main_window: QMainWindow, action_registry: ActionRegistry):
        self.main_window = main_window
        self.actions = action_registry
        self.menubar = main_window.menuBar()
        logger.info("MenuBuilder initialized")
    
    def build_file_menu(self):
        """Build File menu."""
        file_menu = self.menubar.addMenu("&File")
        
        file_menu.addAction(self.actions.get_action("file_new"))
        file_menu.addAction(self.actions.get_action("file_open"))
        
        # Recent files submenu (placeholder)
        recent_menu = file_menu.addMenu("Open &Recent")
        recent_menu.addAction("(No recent files)")
        
        file_menu.addSeparator()
        
        file_menu.addAction(self.actions.get_action("file_save"))
        file_menu.addAction(self.actions.get_action("file_save_as"))
        file_menu.addAction(self.actions.get_action("file_save_all"))
        
        file_menu.addSeparator()
        
        file_menu.addAction(self.actions.get_action("file_close"))
        file_menu.addAction(self.actions.get_action("file_close_all"))
        
        file_menu.addSeparator()
        
        file_menu.addAction(self.actions.get_action("file_exit"))
        
        return file_menu
    
    def build_edit_menu(self):
        """Build Edit menu."""
        edit_menu = self.menubar.addMenu("&Edit")
        
        edit_menu.addAction(self.actions.get_action("edit_undo"))
        edit_menu.addAction(self.actions.get_action("edit_redo"))
        
        edit_menu.addSeparator()
        
        edit_menu.addAction(self.actions.get_action("edit_cut"))
        edit_menu.addAction(self.actions.get_action("edit_copy"))
        edit_menu.addAction(self.actions.get_action("edit_paste"))
        
        edit_menu.addSeparator()
        
        edit_menu.addAction(self.actions.get_action("edit_find"))
        edit_menu.addAction(self.actions.get_action("edit_replace"))
        
        return edit_menu
    
    def build_view_menu(self, dock_manager=None):
        """Build View menu with panel toggles."""
        view_menu = self.menubar.addMenu("&View")
        
        # Panels submenu
        if dock_manager:
            panels_menu = view_menu.addMenu("&Panels")
            
            # Add toggle actions for each registered panel
            for name in dock_manager._panel_registry.keys():
                action_name = f"view_panel_{name}"
                if self.actions.get_action(action_name):
                    panels_menu.addAction(self.actions.get_action(action_name))
        
        view_menu.addSeparator()
        
        view_menu.addAction(self.actions.get_action("view_split_horizontal"))
        view_menu.addAction(self.actions.get_action("view_split_vertical"))
        
        view_menu.addSeparator()
        
        view_menu.addAction(self.actions.get_action("view_reset_layout"))
        
        return view_menu
    
    def build_tools_menu(self):
        """Build Tools menu."""
        tools_menu = self.menubar.addMenu("&Tools")
        
        tools_menu.addAction(self.actions.get_action("tools_settings"))
        tools_menu.addAction(self.actions.get_action("tools_command_palette"))
        
        tools_menu.addSeparator()
        
        tools_menu.addAction(self.actions.get_action("tools_task_manager"))
        tools_menu.addAction(self.actions.get_action("tools_journal"))
        
        return tools_menu
    
    def build_window_menu(self):
        """Build Window menu."""
        window_menu = self.menubar.addMenu("&Window")
        
        window_menu.addAction(self.actions.get_action("window_next_doc"))
        window_menu.addAction(self.actions.get_action("window_prev_doc"))
        
        window_menu.addSeparator()
        
        # Document list submenu (placeholder)
        docs_menu = window_menu.addMenu("&Documents")
        docs_menu.addAction("(No open documents)")
        
        return window_menu
    
    def build_help_menu(self):
        """Build Help menu."""
        help_menu = self.menubar.addMenu("&Help")
        
        help_menu.addAction(self.actions.get_action("help_docs"))
        help_menu.addAction(self.actions.get_action("help_shortcuts"))
        
        help_menu.addSeparator()
        
        help_menu.addAction(self.actions.get_action("help_about"))
        
        return help_menu
    
    def build_all_menus(self, dock_manager=None):
        """Build complete menu bar."""
        self.build_file_menu()
        self.build_edit_menu()
        self.build_view_menu(dock_manager)
        self.build_tools_menu()
        self.build_window_menu()
        self.build_help_menu()
        
        logger.info("Menu bar built successfully")
