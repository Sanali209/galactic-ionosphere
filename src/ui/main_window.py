"""
Main application window with integrated GUI framework.
Uses DockingService for panels, ActionRegistry for actions, MenuBuilder for menus.
"""
from PySide6.QtWidgets import QMainWindow, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from loguru import logger

from .docking import DockingService
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
        self.docking_service = DockingService(self, self.viewmodel.locator)
        self.menu_builder = MenuBuilder(self, self.action_registry)
        
        # Setup UI
        self._setup_central_widget()
        self._register_actions()
        self._build_menus()
        
        # MVVM Binding
        self.viewmodel.statusMessageChanged.connect(self.update_status)
        
        logger.info("MainWindow initialized with GUI framework")
    
    def _setup_central_widget(self):
        """Setup the central widget (will be document area in future)."""
        self.central_widget = QLabel("Document Area\n\nFoundation Template\n\nTry the menus!")
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
            "file_close", "&Close", self._on_file_close, "Ctrl+W")
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
            "view_reset_layout", "&Reset Layout", self._on_reset_layout)
        
        # Tools menu
        self.action_registry.register_action(
            "tools_settings", "&Settings...", self._on_settings, "Ctrl+,")
        self.action_registry.register_action(
            "tools_command_palette", "&Command Palette...", 
            self._on_command_palette, "Ctrl+Shift+P")
        
        # Help menu
        self.action_registry.register_action(
            "help_docs", "&Documentation", self._on_help_docs, "F1")
        self.action_registry.register_action(
            "help_shortcuts", "&Keyboard Shortcuts", self._on_help_shortcuts)
        self.action_registry.register_action(
            "help_about", "&About", self._on_about)
    
    def _build_menus(self):
        """Build all menus."""
        self.menu_builder.build_all_menus()
    
    @Slot(str)
    def update_status(self, msg):
        """Update status message."""
        self.statusBar().showMessage(msg)
    
    def closeEvent(self, event: QCloseEvent):
        """Save state before closing."""
        try:
            # Check if dock manager still exists (not deleted by Qt)
            import shiboken6
            if hasattr(self.docking_service, '_dock_manager') and self.docking_service._dock_manager:
                if shiboken6.isValid(self.docking_service._dock_manager):
                    layout_state = self.docking_service.save_layout()
                    logger.info("Saved layout state")
                else:
                    logger.debug("Dock manager already deleted, skipping layout save")
            else:
                logger.debug("No dock manager available for layout save")
        except Exception as e:
            logger.debug(f"Could not save layout state: {e}")
        
        event.accept()
    
    # =========================================================================
    # Action Handlers - Override in Subclass
    # These methods log a warning by default if not overridden.
    # Override them in your subclass to implement actual functionality.
    # =========================================================================
    
    def _on_file_new(self) -> None: 
        """
        Create a new document.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_file_new() not implemented - override in subclass")
    
    def _on_file_open(self) -> None: 
        """
        Open file dialog and load document.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_file_open() not implemented - override in subclass")
    
    def _on_file_save(self) -> None: 
        """
        Save current document.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_file_save() not implemented - override in subclass")
    
    def _on_file_save_as(self) -> None: 
        """
        Save current document with new name/location.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_file_save_as() not implemented - override in subclass")
    
    def _on_file_close(self) -> None: 
        """
        Close current document.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_file_close() not implemented - override in subclass")
    
    def _on_edit_undo(self) -> None: 
        """
        Undo last action.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_undo() not implemented - override in subclass")
    
    def _on_edit_redo(self) -> None: 
        """
        Redo last undone action.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_redo() not implemented - override in subclass")
    
    def _on_edit_cut(self) -> None: 
        """
        Cut selection to clipboard.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_cut() not implemented - override in subclass")
    
    def _on_edit_copy(self) -> None: 
        """
        Copy selection to clipboard.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_copy() not implemented - override in subclass")
    
    def _on_edit_paste(self) -> None: 
        """
        Paste from clipboard.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_paste() not implemented - override in subclass")
    
    def _on_edit_find(self) -> None: 
        """
        Open find dialog.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_find() not implemented - override in subclass")
    
    def _on_edit_replace(self) -> None: 
        """
        Open find and replace dialog.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_edit_replace() not implemented - override in subclass")
    
    def _on_reset_layout(self) -> None: 
        """
        Reset window layout to default.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_reset_layout() not implemented - override in subclass")
    
    def _on_settings(self) -> None: 
        """
        Open settings dialog.
        
        Default implementation opens the standard SettingsDialog.
        Override to customize settings handling.
        """
        from .settings.settings_dialog import SettingsDialog
        
        dialog = SettingsDialog(self.viewmodel.locator.config, self)
        dialog.exec()
        logger.info("Settings dialog closed")
    
    def _on_command_palette(self) -> None: 
        """
        Open command palette.
        
        Default implementation opens the standard CommandPalette.
        Override to customize command palette handling.
        """
        from .commands.command_palette import CommandPalette
        
        palette = CommandPalette(self.action_registry, self)
        palette.exec()
        logger.info("Command palette closed")
    
    def _on_help_docs(self) -> None: 
        """
        Open documentation.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_help_docs() not implemented - override in subclass")
    
    def _on_help_shortcuts(self) -> None: 
        """
        Show keyboard shortcuts.
        
        Override this method in subclass to implement functionality.
        Default: logs warning and does nothing.
        """
        logger.warning("_on_help_shortcuts() not implemented - override in subclass")
    
    def _on_about(self) -> None: 
        """
        Show about dialog.
        
        Default implementation shows a basic about dialog.
        Override to customize the about information.
        """
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "About", 
                         "Foundation Template\nGUI Framework Demo")

