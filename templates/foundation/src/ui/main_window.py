"""
Main application window with integrated GUI framework.
Uses DockManager for panels, ActionRegistry for actions, MenuBuilder for menus.
"""
from PySide6.QtWidgets import QMainWindow, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from loguru import logger

from .docking.dock_manager import DockManager
from .docking.output_panel import OutputPanel
from .menus.action_registry import ActionRegistry
from .menus.menu_builder import MenuBuilder

class MainWindow(QMainWindow):
    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel
        self.setWindowTitle("Foundation App - GUI Framework Demo")
        self.resize(1200, 800)
        
        # Initialize GUI framework components
        self.action_registry = ActionRegistry(self)
        self.dock_manager = DockManager(self, viewmodel.locator.config)
        self.menu_builder = MenuBuilder(self, self.action_registry)
        
        # Setup UI
        self._setup_central_widget()
        self._register_actions()
        self._register_panels()
        self._build_menus()
        self._restore_state()
        
        # MVVM Binding
        self.viewmodel.statusMessageChanged.connect(self.update_status)
        
        logger.info("MainWindow initialized with GUI framework")
    
    def _setup_central_widget(self):
        """Setup the central widget (will be document area in future)."""
        self.central_widget = QLabel("Document Area\n\nPhase 1 & 2 Complete!\n\nTry the menus!")
        self.central_widget.setAlignment(Qt.AlignCenter)
        self.central_widget.setStyleSheet("font-size: 18px; color: #555;")
        self.setCentralWidget(self.central_widget)
    
    def _register_actions(self):
        """Register all application actions."""
        # File menu
        self.action_registry.register_action(
            "file_new", "&New", self._on_file_new, "Ctrl+N", "Create new document")
        self.action_registry.register_action(
            "file_open", "&Open...", self._on_file_open, "Ctrl+O", "Open file")
        self.action_registry.register_action(
            "file_save", "&Save", self._on_file_save, "Ctrl+S", "Save current file")
        self.action_registry.register_action(
            "file_save_as", "Save &As...", self._on_file_save_as, "Ctrl+Shift+S")
        self.action_registry.register_action(
            "file_save_all", "Save A&ll", self._on_file_save_all)
        self.action_registry.register_action(
            "file_close", "&Close", self._on_file_close, "Ctrl+W")
        self.action_registry.register_action(
            "file_close_all", "Close &All", self._on_file_close_all)
        self.action_registry.register_action(
            "file_exit", "E&xit", self.close, "Alt+F4", "Exit application")
        
        # Edit menu
        self.action_registry.register_action(
            "edit_undo", "&Undo", self._on_edit_undo, "Ctrl+Z")
        self.action_registry.register_action(
            "edit_redo", "&Redo", self._on_edit_redo, "Ctrl+Y")
        self.action_registry.register_action(
            "edit_cut", "Cu&t", self._on_edit_cut, "Ctrl+X")
        self.action_registry.register_action(
            "edit_copy", "&Copy", self._on_edit_copy, "Ctrl+C")
        self.action_registry.register_action(
            "edit_paste", "&Paste", self._on_edit_paste, "Ctrl+V")
        self.action_registry.register_action(
            "edit_find", "&Find...", self._on_edit_find, "Ctrl+F")
        self.action_registry.register_action(
            "edit_replace", "&Replace...", self._on_edit_replace, "Ctrl+H")
        
        # View menu
        self.action_registry.register_action(
            "view_split_horizontal", "Split &Horizontal", 
            self._on_split_horizontal, "Ctrl+\\")
        self.action_registry.register_action(
            "view_split_vertical", "Split &Vertical", 
            self._on_split_vertical, "Ctrl+Shift+\\")
        self.action_registry.register_action(
            "view_reset_layout", "&Reset Layout", self._on_reset_layout)
        
        # Tools menu
        self.action_registry.register_action(
            "tools_settings", "&Settings...", self._on_settings, "Ctrl+,")
        self.action_registry.register_action(
            "tools_command_palette", "&Command Palette...", 
            self._on_command_palette, "Ctrl+Shift+P")
        self.action_registry.register_action(
            "tools_task_manager", "&Task Manager", self._on_task_manager)
        self.action_registry.register_action(
            "tools_journal", "&Journal Viewer", self._on_journal)
        
        # Window menu
        self.action_registry.register_action(
            "window_next_doc", "&Next Document", 
            self._on_next_doc, "Ctrl+Tab")
        self.action_registry.register_action(
            "window_prev_doc", "&Previous Document", 
            self._on_prev_doc, "Ctrl+Shift+Tab")
        
        # Help menu
        self.action_registry.register_action(
            "help_docs", "&Documentation", self._on_help_docs, "F1")
        self.action_registry.register_action(
            "help_shortcuts", "&Keyboard Shortcuts", self._on_help_shortcuts)
        self.action_registry.register_action(
            "help_about", "&About", self._on_about)
    
    def _register_panels(self):
        """Register all panel types."""
        self.dock_manager.register_panel("output", OutputPanel)
        
        # Register panel toggle actions
        self.action_registry.register_action(
            "view_panel_output", "&Output", 
            lambda: self.dock_manager.toggle_panel("output"),
            checkable=True)
    
    def _build_menus(self):
        """Build all menus."""
        self.menu_builder.build_all_menus(self.dock_manager)
    
    def _restore_state(self):
        """Restore window and panel state from config."""
        try:
            config_data = self.viewmodel.locator.config.data
            if hasattr(config_data, 'ui') and hasattr(config_data.ui, 'panels'):
                panel_state = config_data.ui.panels
                if isinstance(panel_state, dict):
                    self.dock_manager.restore_state(panel_state)
                    return
        except Exception as e:
            logger.warning(f"Could not restore panel state: {e}")
        
        # Create default output panel if no saved state
        self.dock_manager.create_panel("output")
    
    @Slot(str)
    def update_status(self, msg):
        """Update status message."""
        self.statusBar().showMessage(msg)
    
    def closeEvent(self, event: QCloseEvent):
        """Save state before closing."""
        try:
            panel_state = self.dock_manager.save_state()
            logger.info(f"Saved state for {len(panel_state)} panels")
            # TODO: Persist to ConfigManager
        except Exception as e:
            logger.error(f"Failed to save panel state: {e}")
        
        event.accept()
    
    # Action handlers (placeholder implementations)
    def _on_file_new(self): 
        logger.info("File > New")
        self.viewmodel.status_message = "New file created (placeholder)"
    
    def _on_file_open(self): 
        logger.info("File > Open")
        self.viewmodel.status_message = "Open dialog (placeholder)"
    
    def _on_file_save(self): 
        logger.info("File > Save")
        self.viewmodel.status_message = "File saved (placeholder)"
    
    def _on_file_save_as(self): 
        logger.info("File > Save As")
    
    def _on_file_save_all(self): 
        logger.info("File > Save All")
    
    def _on_file_close(self): 
        logger.info("File > Close")
    
    def _on_file_close_all(self): 
        logger.info("File > Close All")
    
    def _on_edit_undo(self): 
        logger.info("Edit > Undo")
    
    def _on_edit_redo(self): 
        logger.info("Edit > Redo")
    
    def _on_edit_cut(self): 
        logger.info("Edit > Cut")
    
    def _on_edit_copy(self): 
        logger.info("Edit > Copy")
    
    def _on_edit_paste(self): 
        logger.info("Edit > Paste")
    
    def _on_edit_find(self): 
        logger.info("Edit > Find")
    
    def _on_edit_replace(self): 
        logger.info("Edit > Replace")
    
    def _on_split_horizontal(self): 
        logger.info("View > Split Horizontal")
        self.viewmodel.status_message = "Split functionality coming in Phase 3"
    
    def _on_split_vertical(self): 
        logger.info("View > Split Vertical")
    
    def _on_reset_layout(self): 
        logger.info("View > Reset Layout")
    
    def _on_settings(self): 
        """Open settings dialog."""
        from .settings.settings_dialog import SettingsDialog
        
        dialog = SettingsDialog(self.viewmodel.locator.config, self)
        dialog.exec()
        logger.info("Settings dialog closed")
    
    def _on_command_palette(self): 
        """Open command palette."""
        from .commands.command_palette import CommandPalette
        
        palette = CommandPalette(self.action_registry, self)
        palette.exec()
        logger.info("Command palette closed")
    
    def _on_task_manager(self): 
        logger.info("Tools > Task Manager")
    
    def _on_journal(self): 
        logger.info("Tools > Journal")
    
    def _on_next_doc(self): 
        logger.info("Window > Next Document")
    
    def _on_prev_doc(self): 
        logger.info("Window > Previous Document")
    
    def _on_help_docs(self): 
        logger.info("Help > Documentation")
    
    def _on_help_shortcuts(self): 
        logger.info("Help > Shortcuts")
    
    def _on_about(self): 
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "About", 
                         "Foundation Template\nGUI Framework Demo\n\nPhases 1 & 2 Complete!")
