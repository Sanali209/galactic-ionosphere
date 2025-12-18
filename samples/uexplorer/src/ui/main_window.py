"""
UExplorer Main Window

Directory Opus-inspired dual-pane file manager.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QToolBar, QStatusBar, QLabel
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from loguru import logger


class MainWindow(QMainWindow):
    """
    UExplorer main window with dual-pane layout.
    
    Layout:
    ┌──────────────────────────────────────┐
    │ Menu Bar                             │
    ├──────────────────────────────────────┤
    │ Toolbar                              │
    ├────────┬─────────────────────────────┤
    │ Left   │  Dual File Panes            │
    │ Panel  ├──────────┬──────────────────┤
    │        │ Left     │ Right            │
    │ - Tags │ Pane     │ Pane             │
    │ - Alb. │          │                  │
    ├────────┴──────────┴──────────────────┤
    │ Metadata Panel                       │
    ├──────────────────────────────────────┤
    │ Status Bar                           │
    └──────────────────────────────────────┘
    """
    
    def __init__(self, locator):
        super().__init__()
        
        self.locator = locator
        self.config = locator.config
        
        # Setup window
        self.setup_window()
        self.create_menu_bar()
        self.create_toolbar()
        self.create_central_widget()
        self.create_status_bar()
        
        # Apply theme
        self.apply_theme()
    
    
    from qasync import asyncSlot
    @asyncSlot()
    async def load_initial_roots(self):
        """Load library roots into file panes (call after event loop starts)."""
        from loguru import logger
        logger.info("MainWindow: Triggering initial root load...")
        
        # Trigger refresh on both file panes
        if hasattr(self, 'left_pane'):
            logger.info("  - Loading roots into left pane")
            await self.left_pane.model.refresh_roots()
        
        if hasattr(self, 'right_pane'):
            logger.info("  - Loading roots into right pane")
            await self.right_pane.model.refresh_roots()
            
        # Trigger background scan of all roots
        try:
            from src.ucorefs.discovery.service import DiscoveryService
            discovery = self.locator.get_system(DiscoveryService)
            if discovery:
                logger.info("  - Triggering background library scan...")
                await discovery.scan_all_roots(background=True)
        except Exception as e:
            logger.error(f"Failed to trigger initial scan: {e}")
        
        logger.info("✓ Initial root load triggered")
    
    def closeEvent(self, event):
        """Handle window close event with logging."""
        from loguru import logger
        logger.info("=" * 100)
        logger.info("🚪 WINDOW CLOSE EVENT TRIGGERED")
        logger.info("=" * 100)
        logger.info(f"Event type: {event.type()}")
        logger.info(f"Event spontaneous: {event.spontaneous()}")
        logger.info(f"Window visible before close: {self.isVisible()}")
        logger.info("User is closing the window...")
        logger.info("=" * 100)
        
        # Accept the close event
        event.accept()
        logger.info("Close event accepted, window will close")
    
    def setup_window(self):
        """Setup main window properties."""
        ui_config = self.config.data.ui if hasattr(self.config.data, 'ui') else None
        
        width = ui_config.window_width if ui_config else 1400
        height = ui_config.window_height if ui_config else 900
        
        self.setWindowTitle("UExplorer - UCoreFS File Manager")
        self.resize(width, height)
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Window", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings_dialog)
        edit_menu.addAction(settings_action)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        dual_pane_action = QAction("&Dual Pane", self)
        dual_pane_action.setCheckable(True)
        dual_pane_action.setChecked(True)
        view_menu.addAction(dual_pane_action)
        
        view_menu.addSeparator()
        
        thumbnails_action = QAction("Show &Thumbnails", self)
        thumbnails_action.setCheckable(True)
        thumbnails_action.setChecked(True)
        view_menu.addAction(thumbnails_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        scan_action = QAction("&Scan Directories", self)
        scan_action.setShortcut("F5")
        tools_menu.addAction(scan_action)
        
        tools_menu.addSeparator()
        
        library_action = QAction("&Library Settings...", self)
        library_action.triggered.connect(self.show_library_dialog)  # Connect to handler
        tools_menu.addAction(library_action)
        
        rules_action = QAction("&Rules Manager...", self)
        tools_menu.addAction(rules_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About UExplorer", self)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        scan_btn = QAction("Scan", self)
        scan_btn.setToolTip("Scan directories for changes")
        toolbar.addAction(scan_btn)
        
        toolbar.addSeparator()
        
        tags_btn = QAction("Tags", self)
        tags_btn.setToolTip("Manage tags")
        toolbar.addAction(tags_btn)
        
        albums_btn = QAction("Albums", self)
        albums_btn.setToolTip("Manage albums")
        toolbar.addAction(albums_btn)
        
        toolbar.addSeparator()
        
        rules_btn = QAction("Rules", self)
        rules_btn.setToolTip("Manage automation rules")
        toolbar.addAction(rules_btn)
        
        search_btn = QAction("Search", self)
        search_btn.setToolTip("Advanced search")
        toolbar.addAction(search_btn)
    
    def create_central_widget(self):
        """Create central widget with dual-pane layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (navigation)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right area (dual panes + metadata)
        right_area = self.create_right_area()
        main_splitter.addWidget(right_area)
        
        # Set splitter sizes (20% left, 80% right)
        main_splitter.setSizes([280, 1120])
        
        layout.addWidget(main_splitter)
    
    def create_left_panel(self):
        """Create left navigation panel."""
        from PySide6.QtWidgets import QTabWidget, QTreeWidget, QTreeWidgetItem
        import sys
        from pathlib import Path
        
        # Add widgets to path for imports
        widgets_path = Path(__file__).parent / "widgets"
        if str(widgets_path) not in sys.path:
            sys.path.insert(0, str(widgets_path))
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tabbed navigation
        tabs = QTabWidget()
        
        # Tags tab - use TagTreeWidget
        try:
            from tag_tree import TagTreeWidget
            self.tags_tree = TagTreeWidget(self.locator)
        except Exception as e:
            # Fallback to static tree
            from loguru import logger
            logger.error(f"Failed to load TagTreeWidget: {e}")
            self.tags_tree = QTreeWidget()
            self.tags_tree.setHeaderLabel("Tags")
            self.tags_tree.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
            QTreeWidgetItem(self.tags_tree, [f"Error: {str(e)}"])
        
        tabs.addTab(self.tags_tree, "Tags")
        
        # Albums tab - use AlbumTreeWidget
        try:
            from album_tree import AlbumTreeWidget
            self.albums_tree = AlbumTreeWidget(self.locator)
            self.albums_tree.album_selected.connect(self._on_album_selected)
        except Exception as e:
            logger.warning(f"AlbumTreeWidget not available: {e}")
            self.albums_tree = QTreeWidget()
            self.albums_tree.setHeaderLabel("Albums")
            self.albums_tree.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
            QTreeWidgetItem(self.albums_tree, ["(Error loading albums)"])
        tabs.addTab(self.albums_tree, "Albums")
        
        # Relations tab - use RelationTreeWidget
        try:
            from relation_panel import RelationTreeWidget
            self.relations_tree = RelationTreeWidget(self.locator)
            self.relations_tree.category_selected.connect(self._on_relation_category_selected)
        except Exception as e:
            logger.warning(f"RelationTreeWidget not available: {e}")
            self.relations_tree = QTreeWidget()
            self.relations_tree.setHeaderLabel("Relations")
            self.relations_tree.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
            QTreeWidgetItem(self.relations_tree, ["Duplicates"])
            QTreeWidgetItem(self.relations_tree, ["Similar"])
        tabs.addTab(self.relations_tree, "Relations")
        
        layout.addWidget(tabs)
        
        return panel
    
    def create_right_area(self):
        """Create right area with dual panes and metadata."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical splitter for panes + metadata
        vsplitter = QSplitter(Qt.Vertical)
        
        # Dual file panes
        dual_panes = self.create_dual_panes()
        vsplitter.addWidget(dual_panes)
        
        # Metadata panel
        metadata = self.create_metadata_panel()
        vsplitter.addWidget(metadata)
        
        # Set sizes (70% panes, 30% metadata)
        vsplitter.setSizes([630, 270])
        
        layout.addWidget(vsplitter)
        
        return widget
    
    def create_dual_panes(self):
        """Create dual file browser panes."""
        splitter = QSplitter(Qt.Horizontal)
        
        # Import FilePaneWidget
        import sys
        from pathlib import Path
        widgets_path = Path(__file__).parent / "widgets"
        if str(widgets_path) not in sys.path:
            sys.path.insert(0, str(widgets_path))
        
        from file_pane import FilePaneWidget
        from metadata_panel import MetadataPanel
        
        # Left pane (qasync handles event loop globally)
        self.left_pane = FilePaneWidget(self.locator)
        self.left_pane.selection_changed.connect(self.on_selection_changed)
        splitter.addWidget(self.left_pane)
        
        # Right pane  
        self.right_pane = FilePaneWidget(self.locator)
        self.right_pane.selection_changed.connect(self.on_selection_changed)
        splitter.addWidget(self.right_pane)
        
        # Equal sizes
        splitter.setSizes([560, 560])
        
        return splitter
    
    def on_selection_changed(self, record_ids):
        """Handle selection change (any pane)."""
        if not record_ids:
            self.metadata_panel.clear()
            self.status_label.setText("Ready")
            return
            
        self.status_label.setText(f"{len(record_ids)} item(s) selected")
        
        # Update metadata panel (async fetch)
        import asyncio
        asyncio.ensure_future(self._update_metadata(record_ids[0]))

    async def _update_metadata(self, record_id_str):
        """Fetch record and update panel."""
        try:
            from bson import ObjectId
            from src.ucorefs.models.file_record import FileRecord
            
            # Fetch fresh from DB (to get tags etc)
            record = await FileRecord.get(ObjectId(record_id_str))
            
            # If not file, maybe directory? MetadataPanel handles FileRecord primarily.
            # DirectoryRecord support later.
            if record:
                self.metadata_panel.set_file(record)
            else:
               self.metadata_panel.clear()
               
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to update metadata: {e}")
    
    def on_left_selection_changed(self, record_ids):
        # Deprecated
        self.on_selection_changed(record_ids)
    
    def on_right_selection_changed(self, record_ids):
        # Deprecated
        self.on_selection_changed(record_ids)
    
    def create_metadata_panel(self):
        """Create metadata editor panel."""
        import sys
        from pathlib import Path
        widgets_path = Path(__file__).parent / "widgets"
        if str(widgets_path) not in sys.path:
            sys.path.insert(0, str(widgets_path))
            
        from metadata_panel import MetadataPanel
        self.metadata_panel = MetadataPanel(self.locator)
        return self.metadata_panel
    
    def create_status_bar(self):
        """Create status bar."""
        status = QStatusBar()
        self.setStatusBar(status)
        
        # Status message
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label)
        
        # File count
        self.file_count_label = QLabel("0 files")
        status.addPermanentWidget(self.file_count_label)
        
        # Database status
        self.db_status_label = QLabel("⚠ Not Connected")
        status.addPermanentWidget(self.db_status_label)
    
    def show_library_dialog(self):
        """Show library settings dialog."""
        import sys
        from pathlib import Path
        
        # Add dialogs to path
        dialogs_path = Path(__file__).parent / "dialogs"
        if str(dialogs_path) not in sys.path:
            sys.path.insert(0, str(dialogs_path))
        
        from library_dialog import LibraryDialog
        dialog = LibraryDialog(self.locator, parent=self)
        dialog.exec()
        
        # Refresh file models after dialog closes
        import asyncio
        asyncio.ensure_future(self.left_pane.model.refresh_roots())
        asyncio.ensure_future(self.right_pane.model.refresh_roots())
    
    def show_settings_dialog(self):
        """Show settings dialog."""
        try:
            from src.core.config import ConfigManager
            from src.ui.settings.settings_dialog import SettingsDialog
            
            config = self.locator.get_system(ConfigManager)
            dialog = SettingsDialog(config, parent=self)
            dialog.exec()
            
            logger.info("Settings dialog closed")
        except Exception as e:
            logger.error(f"Failed to open settings dialog: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Could not open settings: {e}")
    
    def _on_album_selected(self, album_id: str, is_smart: bool, query: dict):
        """Handle album selection - filter files."""
        logger.info(f"Album selected: {album_id}, smart={is_smart}, query={query}")
        self.status_label.setText(f"Viewing album (smart={is_smart})")
        # TODO: Filter file panes based on album contents
    
    def _on_relation_category_selected(self, category: str):
        """Handle relation category selection."""
        logger.info(f"Relation category selected: {category}")
        self.status_label.setText(f"Viewing {category} relations")
    
    def apply_theme(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 4px;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #0e639c;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 3px;
                padding: 4px;
            }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
            }
            QLabel {
                color: #cccccc;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
            }
        """)
