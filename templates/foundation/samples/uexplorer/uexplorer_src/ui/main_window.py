"""
UExplorer Main Window

Directory Opus-inspired dual-pane file manager.
Now powered by PySide6-QtAds for professional docking!
"""
import sys
from pathlib import Path

# Add foundation to path
# Path: samples/uexplorer/uexplorer_src/ui/main_window.py
# Go up 4 levels to get to foundation root
foundation_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(foundation_path))

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QToolBar, QStatusBar, QLabel
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from loguru import logger

# Import DockingService
from src.ui.documents.docking_service import DockingService


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
    
    def __init__(self, viewmodel):
        """
        Initialize UExplorer main window.
        
        Args:
            viewmodel: MainViewModel instance (contains locator reference)
        """
        super().__init__()
        
        # Extract locator from viewmodel (run_app passes viewmodel)
        self.viewmodel = viewmodel
        self.locator = viewmodel.locator
        self.config = self.locator.config
        
        # Initialize centralized UI managers
        self._init_managers()
        
        # Setup window
        self.setup_window()
        
        # Initialize ActionRegistry BEFORE creating menus
        self.setup_action_registry()
        
        self.create_menu_bar()
        self.create_toolbar()
        self.create_central_widget()
        self.create_status_bar()
        
        # Setup NEW docking system (PySide6-QtAds)
        self.setup_docking_service()
        
        # Connect TaskSystem to UI progress
        self.setup_task_progress()
        
        # Apply theme
        self.apply_theme()
        
        # Restore docking layout from saved state
        self.restore_docking_layout()
        logger.info("Layout persistence enabled")
    
    def _init_managers(self):
        """Initialize centralized UI managers."""
        from uexplorer_src.ui.managers import FilterManager, SelectionManager
        
        # Create managers
        self.filter_manager = FilterManager(self)
        self.selection_manager = SelectionManager(self)
        
        # Create MVVM components
        from uexplorer_src.viewmodels import DocumentManager, SearchPipeline
        self.document_manager = DocumentManager(self)
        self.search_pipeline = SearchPipeline(self.locator, self)
        
        # Connect search pipeline to document manager
        self.search_pipeline.search_completed.connect(
            self.document_manager.send_results_to_active
        )
        
        # Store in viewmodel for easy access from child widgets
        self.viewmodel.filter_manager = self.filter_manager
        self.viewmodel.selection_manager = self.selection_manager
        self.viewmodel.document_manager = self.document_manager
        
        logger.info("UI managers initialized (FilterManager, SelectionManager, DocumentManager)")
    
    
    from qasync import asyncSlot
    @asyncSlot()
    async def load_initial_roots(self):
        """Load library roots into file panes (call after event loop starts)."""
        from loguru import logger
        logger.info("MainWindow: Triggering initial root load...")
        
        # Refresh all panes in split area
        if hasattr(self, 'split_area'):
            for pane in self.split_area.get_all_panes():
                logger.info(f"  - Loading roots into pane: {pane.title}")
                if hasattr(pane.file_pane, 'model'):
                    await pane.file_pane.model.refresh_roots()
        
        # Fallback for old-style panes
        elif hasattr(self, 'left_pane') and self.left_pane:
            logger.info("  - Loading roots into left pane")
            await self.left_pane.model.refresh_roots()
            
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
        
        # Save layout state before closing
        self.save_layout()
        
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
    
    def setup_action_registry(self):
        """Initialize ActionRegistry with all UExplorer actions."""
        from src.ui.menus.action_registry import ActionRegistry
        from uexplorer_src.ui.actions.action_definitions import register_all_actions
        
        # Create action registry
        self.action_registry = ActionRegistry(self)
        
        # Register all actions
        register_all_actions(self.action_registry, self)
        
        logger.info("ActionRegistry initialized with UExplorer actions")
    
    def create_menu_bar(self):
        """Create menu bar using Foundation's MenuBuilder."""
        from src.ui.menus.menu_builder import MenuBuilder
        
        # Use MenuBuilder to construct standard menus
        menu_builder = MenuBuilder(self, self.action_registry)
        
        # Build standard menus (File, Edit, View, Tools, Window, Help)
        # Note: MenuBuilder provides standard structure, we customize below
        self._build_custom_menus()
        
        logger.info("Menu bar created with ActionRegistry")
    
    def _build_custom_menus(self):
        """Build UExplorer-specific menu structure."""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_registry.get_action("file.new_window"))
        file_menu.addAction(self.action_registry.get_action("file.new_browser"))
        file_menu.addSeparator()
        file_menu.addAction(self.action_registry.get_action("file.exit"))
        
        # EDIT MENU
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_registry.get_action("edit.settings"))
        
        # VIEW MENU
        view_menu = menubar.addMenu("&View")
        
        # Panels submenu
        panels_menu = view_menu.addMenu("&Panels")
        panels_menu.addAction(self.action_registry.get_action("view.panel.tags"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.albums"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.relations"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.properties"))
        panels_menu.addSeparator()
        panels_menu.addAction(self.action_registry.get_action("view.panel.filters"))
        panels_menu.addAction(self.action_registry.get_action("view.panel.search"))
        
        view_menu.addSeparator()
        
        # Split actions
        view_menu.addAction(self.action_registry.get_action("view.split_horizontal"))
        view_menu.addAction(self.action_registry.get_action("view.split_vertical"))
        view_menu.addAction(self.action_registry.get_action("view.close_split"))
        
        view_menu.addSeparator()
        view_menu.addAction(self.action_registry.get_action("view.reset_layout"))
        
        view_menu.addSeparator()
        view_menu.addAction(self.action_registry.get_action("view.thumbnails"))
        
        # TOOLS MENU
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.action_registry.get_action("tools.scan"))
        tools_menu.addSeparator()
        tools_menu.addAction(self.action_registry.get_action("tools.library"))
        tools_menu.addAction(self.action_registry.get_action("tools.rules"))
        tools_menu.addSeparator()
        tools_menu.addAction(self.action_registry.get_action("tools.command_palette"))
        
        # HELP MENU
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.action_registry.get_action("help.shortcuts"))
        help_menu.addSeparator()
        help_menu.addAction(self.action_registry.get_action("help.about"))
    
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
        
        self.toolbar = toolbar
        
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
        """Create the central widget area with file browser."""
        # Create the split container with drag & drop support
        import sys
        from pathlib import Path
        
        # Add Foundation to path
        foundation_path = Path(__file__).parent.parent.parent.parent / "templates" / "foundation"
        if str(foundation_path) not in sys.path:
            sys.path.insert(0, str(foundation_path))
        
        from src.ui.documents.split_manager import SplitManager
        from src.ui.documents.split_container import SplitContainer
        
        # Import local file_pane_document
        docs_path = Path(__file__).parent / "documents"
        if str(docs_path) not in sys.path:
            sys.path.insert(0, str(docs_path))
        
        from file_pane_document import FilePaneDocument
        
        self.split_manager = SplitManager()
        
        # Create initial file browser pane as a document
        initial_pane = FilePaneDocument(self.locator, "Browser 1")
        
        # Get root container and add document
        root_container = SplitContainer(self.split_manager.root.id)
        root_container.add_document(initial_pane, "Browser 1")
        
        # Store container widget reference AND connect to drag coordinator
        self.split_manager.set_container_widget(self.split_manager.root.id, root_container)
        
        # Set as central widget directly
        self.setCentralWidget(root_container)
        
        # Connect signals from root container
        root_container.document_activated.connect(self.on_selection_changed)
        
        logger.info("✓ Central widget setup complete with drag & drop enabled")
    
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
        
        # File panes (no more embedded metadata - now in dock panel)
        dual_panes = self.create_dual_panes()
        layout.addWidget(dual_panes)
        
        return widget
    
    def create_dual_panes(self):
        """Create split document area for file browsing."""
        import sys
        from pathlib import Path
        
        # Add documents to path
        docs_path = Path(__file__).parent / "documents"
        if str(docs_path) not in sys.path:
            sys.path.insert(0, str(docs_path))
        # Central widget - File browser with Foundation's SplitManager (drag & drop enabled)
        import sys
        from pathlib import Path
        
        # Add Foundation to path
        foundation_path = Path(__file__).parent.parent.parent.parent / "templates" / "foundation"
        if str(foundation_path) not in sys.path:
            sys.path.insert(0, str(foundation_path))
        
        from src.ui.documents.split_manager import SplitManager, SplitOrientation
        from src.ui.documents.split_container import SplitContainer
        
        # Import local file_pane_document
        docs_path = Path(__file__).parent / "documents"
        if str(docs_path) not in sys.path:
            sys.path.insert(0, str(docs_path))
        
        from file_pane_document import FilePaneDocument
        
        self.split_manager = SplitManager()
        
        # Create initial file browser pane as a document
        initial_pane = FilePaneDocument(self.locator, "Browser 1")
        
        # Get root container and add document
        root_container = SplitContainer(self.split_manager.root.id)
        root_container.add_document(initial_pane, "Browser 1")
        
        # Store container widget reference
        self.split_manager.root.container_widget = root_container
        
        # Set as central widget
        self.setCentralWidget(root_container)
        
        logger.info("✓ Central widget setup complete with drag & drop enabled")
        
        # Connect signals from root container
        root_container.document_activated.connect(self.on_selection_changed)
        
        # Store reference
        self.central_container = root_container
        
        return root_container
    
    def on_selection_changed(self, record_ids):
        """Handle selection change (any pane)."""
        if not record_ids:
            self._clear_metadata()
            self.status_label.setText("Ready")
            return
            
        self.status_label.setText(f"{len(record_ids)} item(s) selected")
        
        # Update metadata panel (async fetch)
        import asyncio
        asyncio.ensure_future(self._update_metadata(record_ids[0]))

    def _clear_metadata(self):
        """Clear metadata panel if available."""
        if hasattr(self, 'properties_panel') and self.properties_panel:
            mp = self.properties_panel.metadata_panel
            if mp:
                mp.clear()

    async def _update_metadata(self, record_id_str):
        """Fetch record and update panel."""
        try:
            from bson import ObjectId
            from src.ucorefs.models.file_record import FileRecord
            
            # Fetch fresh from DB (to get tags etc)
            record = await FileRecord.get(ObjectId(record_id_str))
            
            # Get metadata panel from docked properties panel
            if hasattr(self, 'properties_panel') and self.properties_panel:
                mp = self.properties_panel.metadata_panel
                if record and mp:
                    mp.set_file(record)
                elif mp:
                    mp.clear()
               
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
        
        # Progress bar (hidden by default)
        from PySide6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        status.addPermanentWidget(self.progress_bar)
    
    def setup_task_progress(self):
        """
        TaskSystem integration note.
        
        Foundation's TaskSystem manages tasks via database (TaskRecord),
        not Qt signals. UI progress tracking would require:
        1. Polling TaskRecord.find({"status": "running"})
        2. Custom signal emission in task handlers
        3. Or use DiscoveryService progress callbacks
        
        For demonstration, we verify TaskSystem is available.
        """
        from src.core.tasks.system import TaskSystem
        
        try:
            task_system = self.locator.get_system(TaskSystem)
            if not task_system:
                logger.warning("TaskSystem not available")
                return
            
            logger.info("TaskSystem available - tasks tracked in database")
            # Note: DiscoveryService already uses TaskSystem internally
            # and provides its own progress updates via scan completion
            
        except Exception as e:
            logger.error(f"Failed to setup task progress: {e}")
    
    def show_progress(self, visible: bool, value: int = 0, maximum: int = 100):
        """Show/hide progress bar in status bar."""
        self.progress_bar.setVisible(visible)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
    
    def update_progress(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
    
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
        # Use file_pane_left/right (set by docking service) not left_pane/right_pane
        if hasattr(self, 'file_pane_left') and self.file_pane_left and self.file_pane_left.model:
            asyncio.ensure_future(self.file_pane_left.model.refresh_roots())
        if hasattr(self, 'file_pane_right') and self.file_pane_right and self.file_pane_right.model:
            asyncio.ensure_future(self.file_pane_right.model.refresh_roots())
    
    def show_settings_dialog(self):
        """Show settings dialog."""
        try:
            from src.core.config import ConfigManager
            from src.ui.settings.settings_dialog import SettingsDialog
            
            # ConfigManager is available via locator.config, not as a registered system
            config = self.locator.config
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
    
    def setup_docking_service(self):
        """Initialize NEW docking system with DockingService (PySide6-QtAds)."""
        logger.info("Setting up DockingService (PySide6-QtAds)...")
        
        # Create docking service
        self.docking_service = DockingService(self)
        
        # Connect document activation to DocumentManager
        self.docking_service.document_activated.connect(self._on_document_tab_activated)
        
        # Create file browser documents (center area, tabbed)
        self._create_file_documents()
        
        # Create tool panels (sides)
        self._create_tool_panels()
        
        logger.info("✓ DockingService initialized successfully!")
    
    def _on_document_tab_activated(self, doc_id: str):
        """Handle document tab activation - update DocumentManager."""
        if hasattr(self, 'document_manager'):
            self.document_manager.set_active(doc_id)
            logger.info(f"Active document switched to: {doc_id}")
    
    def _create_file_documents(self):
        """Create file browser documents (restores previous session)."""
        import sys
        import json
        from pathlib import Path
        widgets_path = Path(__file__).parent / "widgets"
        if str(widgets_path) not in sys.path:
            sys.path.insert(0, str(widgets_path))
        from file_pane import FilePaneWidget
        
        # Load browser count from previous session
        browser_count = 1  # Default: 1 browser
        session_file = Path(__file__).parent.parent.parent / "session.json"
        if session_file.exists():
            try:
                session_data = json.loads(session_file.read_text())
                browser_count = session_data.get("browser_count", 1)
                logger.debug(f"Session restore: {browser_count} browsers")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")
        
        # Create browsers from saved session
        for i in range(1, browser_count + 1):
            doc_id = f"browser_{i}"
            pane = FilePaneWidget(self.locator)
            self.docking_service.add_document(
                doc_id=doc_id,
                widget=pane,
                title=f"Browser {i}",
                area="center",
                closable=True
            )
            
            # Create ViewModel and register with DocumentManager
            if hasattr(self, 'document_manager'):
                vm = self.document_manager.create_document(doc_id)
                pane.set_viewmodel(vm)
            
            # Connect to managers
            self._connect_pane_to_managers(pane)
            
            # Keep first as file_pane_left for compatibility
            if i == 1:
                self.file_pane_left = pane
        
        logger.info(f"✓ Created {browser_count} browser(s) (Ctrl+T for more)")
    
    def _connect_pane_to_managers(self, pane):
        """Connect a file pane to FilterManager and SelectionManager."""
        if hasattr(self, 'filter_manager') and hasattr(self, 'selection_manager'):
            pane.set_managers(
                filter_manager=self.filter_manager,
                selection_manager=self.selection_manager
            )
            # Connect Find Similar to image search
            pane.find_similar.connect(self._on_find_similar_image)
            logger.debug(f"Connected pane to managers")
    
    def _connect_panes_to_managers(self):
        """Legacy compatibility - connect file_pane_left if exists."""
        if hasattr(self, 'file_pane_left') and self.file_pane_left:
            self._connect_pane_to_managers(self.file_pane_left)
    
    def _create_tool_panels(self):
        """Create tool panels from existing panel classes."""
        import sys
        from pathlib import Path
        docking_path = Path(__file__).parent / "docking"
        if str(docking_path) not in sys.path:
            sys.path.insert(0, str(docking_path))
        
        from tag_panel import TagPanel
        from album_panel import AlbumPanel
        from properties_panel import PropertiesPanel
        from relations_panel import RelationsPanel
        
        # TAGS PANEL (Left)
        self.tags_panel = TagPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="tags",
            widget=self.tags_panel,
            title="Tags",
            area="left",
            closable=False
        )
        
        # ALBUMS PANEL (Left, below tags)
        self.albums_panel = AlbumPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="albums",
            widget=self.albums_panel,
            title="Albums",
            area="left",
            closable=False
        )
        
        # FILTER PANEL (Left, below albums)
        from uexplorer_src.ui.widgets.filter_panel import FilterPanel
        self.filter_panel = FilterPanel(
            filter_manager=self.filter_manager if hasattr(self, 'filter_manager') else None,
            locator=self.locator
        )
        self.docking_service.add_panel(
            panel_id="filters",
            widget=self.filter_panel,
            title="Filters",
            area="left",
            closable=True
        )
        
        # SEARCH PANEL (Left, below filters)
        from uexplorer_src.ui.docking.search_dock_panel import SearchDockPanel
        self.search_dock_panel = SearchDockPanel(
            filter_manager=self.filter_manager if hasattr(self, 'filter_manager') else None,
            locator=self.locator
        )
        self.docking_service.add_panel(
            panel_id="search",
            widget=self.search_dock_panel,
            title="Search",
            area="left",
            closable=True
        )
        
        # PROPERTIES PANEL (Right)
        self.properties_panel = PropertiesPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="properties",
            widget=self.properties_panel,
            title="Properties",
            area="right",
            closable=False
        )
        
        # RELATIONS PANEL (Bottom)
        self.relations_panel = RelationsPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="relations",
            widget=self.relations_panel,
            title="Related Files",
            area="bottom",
            closable=False
        )
        
        # Connect panel signals (keep existing connections)
        if hasattr(self.tags_panel, 'tree'):
            self.tags_panel.tree.files_dropped_on_tag.connect(
                lambda tag_id, files: logger.info(f"Tagged {len(files)} files with {tag_id}")
            )
        
        if hasattr(self.albums_panel, 'tree'):
            self.albums_panel.tree.album_selected.connect(self._on_album_selected)
        
        if hasattr(self.relations_panel, 'tree'):
            self.relations_panel.tree.category_selected.connect(self._on_relation_category_selected)
        
        # Connect PropertiesPanel to SelectionManager for auto-update
        if hasattr(self, 'selection_manager') and hasattr(self, 'properties_panel'):
            self.selection_manager.active_changed.connect(self._on_active_file_changed)
        
        # Connect SearchDockPanel to execute search via pipeline
        self.search_dock_panel.search_requested.connect(self._on_search_requested)
        logger.info("Connected SearchDockPanel.search_requested to _on_search_requested")
        
        # Connect FilterPanel to auto-refresh on filter apply
        if hasattr(self, 'filter_panel'):
            self.filter_panel.filters_applied.connect(self._on_filters_applied)
        
        logger.info("✓ Created all tool panels")
    
    def _on_active_file_changed(self, file_id):
        """Update properties panel when active file changes."""
        if file_id and hasattr(self, 'properties_panel'):
            self.properties_panel.set_file(str(file_id))
    
    def _on_search_requested(self, mode: str, query: str, fields: list):
        """
        Handle search request using SearchPipeline.
        
        Results automatically go to active document via DocumentManager.
        """
        import asyncio
        asyncio.ensure_future(self._perform_search(mode, query, fields))
    
    async def _perform_search(self, mode: str, query_text: str, fields: list = None):
        """Execute search via SearchPipeline."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        
        if fields is None:
            fields = ["name", "path"]
        
        logger.info(f"Search: mode={mode}, query='{query_text}', fields={fields}")
        
        # Build SearchQuery
        search_query = SearchQuery(
            text=query_text,
            mode="vector" if mode == "vector" else "text",
            fields=fields,
            limit=100
        )
        
        # Add filters from filter_manager
        if hasattr(self, 'filter_manager') and self.filter_manager.is_active():
            search_query.filters = {
                "file_type": self.filter_manager.get_selected_types() if hasattr(self.filter_manager, 'get_selected_types') else [],
                "rating": self.filter_manager.get_min_rating() if hasattr(self.filter_manager, 'get_min_rating') else 0,
            }
        
        # Execute via pipeline (results auto-sent to active document)
        if hasattr(self, 'search_pipeline'):
            await self.search_pipeline.execute(search_query)
    
    def _on_find_similar_image(self, file_id):
        """Handle Find Similar request - execute image→vector search."""
        import asyncio
        asyncio.ensure_future(self._perform_image_search(file_id))
    
    async def _perform_image_search(self, file_id):
        """Execute image similarity search via SearchPipeline."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        from bson import ObjectId
        
        logger.info(f"Find Similar: image_id={file_id}")
        
        # Build image search query
        search_query = SearchQuery(
            mode="image",
            file_id=ObjectId(file_id) if isinstance(file_id, str) else file_id,
            limit=50
        )
        
        # Execute via pipeline (results auto-sent to active document)
        if hasattr(self, 'search_pipeline'):
            await self.search_pipeline.execute(search_query)
    
    def _on_search_requested(self, mode: str, query: str, fields: list):
        """Handle search requested from SearchDockPanel."""
        logger.info(f">>> _on_search_requested CALLED: mode={mode}, query='{query}', fields={fields}")
        import asyncio
        asyncio.ensure_future(self._execute_search(mode, query, fields))
    
    async def _execute_search(self, mode: str, query: str, fields: list):
        """Execute search with given parameters."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        
        logger.info(f"Search requested: mode={mode}, query='{query}', fields={fields}")
        
        search_query = SearchQuery(
            text=query,
            mode=mode,
            fields=fields if fields else ["name"],
            limit=100
        )
        
        if hasattr(self, 'search_pipeline'):
            await self.search_pipeline.execute(search_query)
    
    def _on_filters_applied(self):
        """Handle filters applied - execute filter-only search."""
        import asyncio
        asyncio.ensure_future(self._perform_filter_search())
    
    async def _perform_filter_search(self):
        """Execute search with only filter conditions (no text query)."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        
        logger.info("Filters applied - refreshing results")
        
        # Build filter query
        filters = {}
        if hasattr(self, 'filter_manager') and self.filter_manager.is_active():
            filters = {
                "file_type": self.filter_manager.get_selected_types() if hasattr(self.filter_manager, 'get_selected_types') else [],
                "rating": self.filter_manager.get_min_rating() if hasattr(self.filter_manager, 'get_min_rating') else 0,
            }
        
        search_query = SearchQuery(
            text="",  # No text query
            mode="text",
            filters=filters,
            limit=100
        )
        
        if hasattr(self, 'search_pipeline'):
            await self.search_pipeline.execute(search_query)
    
    def _toggle_panel(self, panel_name: str):
        """Toggle panel visibility."""
        if hasattr(self, 'docking_service'):
            self.docking_service.toggle_panel(panel_name)
    
    def save_layout(self):
        """Save docking layout state to config."""
        if not hasattr(self, 'docking_service'):
            logger.warning("No docking_service to save layout from")
            return
        
        try:
            from pathlib import Path
            
            # Get layout state as bytes
            layout_state = self.docking_service.save_layout()
            
            # Save to file in config directory
            layout_file = Path(__file__).parent.parent.parent / "docking_layout.bin"
            layout_file.write_bytes(layout_state)
            
            # Save browser count for session restore
            if hasattr(self, 'document_manager'):
                import json
                session_file = Path(__file__).parent.parent.parent / "session.json"
                browser_count = len([k for k in self.document_manager.documents if k.startswith("browser_")])
                session_data = {"browser_count": browser_count}
                session_file.write_text(json.dumps(session_data))
                logger.debug(f"Session saved: {browser_count} browsers")
            
            logger.info(f"✓ Docking layout saved to {layout_file}")
        except Exception as e:
            logger.error(f"Failed to save docking layout: {e}")
    
    def restore_docking_layout(self):
        """Restore docking layout from saved state."""
        if not hasattr(self, 'docking_service'):
            logger.warning("No docking_service to restore layout to")
            return False
        
        try:
            from pathlib import Path
            
            layout_file = Path(__file__).parent.parent.parent / "docking_layout.bin"
            
            if layout_file.exists():
                layout_state = layout_file.read_bytes()
                self.docking_service.restore_layout(layout_state)
                logger.info(f"✓ Docking layout restored from {layout_file}")
                return True
            else:
                logger.info("No saved docking layout found, using default")
                return False
        except Exception as e:
            logger.error(f"Failed to restore docking layout: {e}")
            return False
    
    def _split_horizontal(self):
        """Split current pane horizontally (side by side) - Now with drag & drop!"""
        from src.ui.documents.split_manager import SplitOrientation
        from PySide6.QtWidgets import QSplitter
        from PySide6.QtCore import Qt
        
        import sys
        from pathlib import Path
        docs_path = Path(__file__).parent / "documents"
        if str(docs_path) not in sys.path:
            sys.path.insert(0, str(docs_path))
        from file_pane_document import FilePaneDocument
        
        if hasattr(self, 'split_manager'):
            # Always split the root - it's the only container we have
            root_id = self.split_manager.root.id
            
            # Only split if root is still a container (hasn't been split yet)
            if self.split_manager.root.is_container:
                new_id = self.split_manager.split_node(root_id, SplitOrientation.HORIZONTAL)
                
                if new_id:
                    # Get both containers (the root now has two children after split)
                    root = self.split_manager.root
                    left_node = root.children[0]
                    right_node = root.children[1]
                    
                    # Create new file pane for the right container
                    new_pane = FilePaneDocument(self.locator, "Browser 2")
                    
                    # Create split container for the right side
                    from src.ui.documents.split_container import SplitContainer
                    right_container = SplitContainer(right_node.id)
                    right_container.add_document(new_pane, new_pane.title)
                    self.split_manager.set_container_widget(right_node.id, right_container)
                    
                    # Create QSplitter to hold both containers
                    splitter = QSplitter(Qt.Horizontal)
                    splitter.addWidget(left_node.container_widget)  # Left keeps the existing container
                    splitter.addWidget(right_container)
                    
                    # Replace central widget with splitter
                    self.setCentralWidget(splitter)
                    
                    logger.info(f"Split horizontal - Browser 2 created with drag & drop")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Already Split",
                    "View is already split! Use drag & drop to rearrange panes."
                )
    
    def _split_vertical(self):
        """Split current pane vertically (top/bottom) - Now with drag & drop!"""
        from src.ui.documents.split_manager import SplitOrientation
        from PySide6.QtWidgets import QSplitter
        from PySide6.QtCore import Qt
        
        import sys
        from pathlib import Path
        docs_path = Path(__file__).parent / "documents"
        if str(docs_path) not in sys.path:
            sys.path.insert(0, str(docs_path))
        from file_pane_document import FilePaneDocument
        
        if hasattr(self, 'split_manager'):
            # Always split the root - it's the only container we have  
            root_id = self.split_manager.root.id
            
            # Only split if root is still a container (hasn't been split yet)
            if self.split_manager.root.is_container:
                new_id = self.split_manager.split_node(root_id, SplitOrientation.VERTICAL)
                
                if new_id:
                    # Get both containers
                    root = self.split_manager.root
                    top_node = root.children[0]
                    bottom_node = root.children[1]
                    
                    # Create new file pane for the bottom container
                    new_pane = FilePaneDocument(self.locator, "Browser 2")
                    
                    # Create split container for the bottom side
                    from src.ui.documents.split_container import SplitContainer
                    bottom_container = SplitContainer(bottom_node.id)
                    bottom_container.add_document(new_pane, new_pane.title)
                    self.split_manager.set_container_widget(bottom_node.id, bottom_container)
                    
                    # Create QSplitter to hold both containers
                    splitter = QSplitter(Qt.Vertical)
                    splitter.addWidget(top_node.container_widget)
                    splitter.addWidget(bottom_container)
                    
                    # Replace central widget with splitter
                    self.setCentralWidget(splitter)
                    
                    logger.info(f"Split vertical - Browser 2 created with drag & drop")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Already Split",
                    "View is already split! Use drag & drop to rearrange panes."
                )
    
    def _close_split(self):
        """Close current split and merge with sibling."""
        logger.info("Close split requested (not yet fully implemented for new system)")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Close Split",
            "Close split functionality coming for new drag & drop system!\n\n"
            "For now, you can rearrange panes by dragging tabs."
        )
    
    def new_window(self):
        """Open a new UExplorer window."""
        logger.info("New window requested (not yet implemented)")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "New Window",
            "Multiple window support coming soon!"
        )
    
    def show_command_palette(self):
        """Show command palette dialog."""
        from src.ui.commands.command_palette import CommandPalette
        
        palette = CommandPalette(self.action_registry, self)
        palette.exec()
        
        logger.info("Command palette shown")
    
    def show_shortcuts_dialog(self):
        """Show keyboard shortcuts dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setMinimumSize(500, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("<h2>⌨️ Keyboard Shortcuts</h2>")
        layout.addWidget(title)
        
        # Get all shortcuts from ActionRegistry
        shortcuts = []
        for action_name in self.action_registry._actions.keys():
            action = self.action_registry.get_action(action_name)
            shortcut = action.shortcut().toString() if action.shortcut().toString() else ""
            if shortcut:  # Only show actions with shortcuts
                shortcuts.append((shortcut, action.text().replace("&", "")))
        
        # Sort by shortcut
        shortcuts.sort(key=lambda x: x[0])
        
        # Table
        table = QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        for row, (shortcut, action_text) in enumerate(shortcuts):
            table.setItem(row, 0, QTableWidgetItem(shortcut))
            table.setItem(row, 1, QTableWidgetItem(action_text))
        
        layout.addWidget(table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_about_dialog(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        
        QMessageBox.about(
            self,
            "About UExplorer",
            "<h2>UExplorer</h2>"
            "<p>Universal File Explorer with AI Features</p>"
            "<p><b>Version:</b> 0.9.10</p>"
            "<p><b>Built with:</b></p>"
            "<ul>"
            "<li>PySide6 (Qt)</li>"
            "<li>MongoDB (Beanie ORM)</li>"
            "<li>ChromaDB (Vector Search)</li>"
        )
    
    
    # Old save_layout removed - using DockingService version at line ~708
    # Docking layout is now saved in docking_layout.bin
    
    def reset_layout(self):
        """Reset layout to defaults."""
        from loguru import logger
        import json
        from pathlib import Path
        
        try:
            # Delete saved layout file
            layout_file = Path(__file__).parent.parent.parent / "layout.json"
            if layout_file.exists():
                layout_file.unlink()
                logger.info("Layout file deleted")
            
            # Reset dock panels to default positions
            if hasattr(self, 'dock_manager'):
                # Show all panels
                for name in ['tags', 'albums', 'relations', 'properties']:
                    self.dock_manager.show_panel(name)
            
            # Note: Split area would need app restart for full reset
            logger.info("Layout reset to defaults. Restart for full effect.")
        except Exception as e:
            logger.error(f"Failed to reset layout: {e}")
    
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
    
    def show_rules_dialog(self):
        """Show the Rule Manager dialog."""
        from uexplorer_src.ui.dialogs.rule_manager_dialog import RuleManagerDialog
        
        dialog = RuleManagerDialog(locator=self.locator, parent=self)
        dialog.exec()
        logger.info("Rule Manager dialog closed")
    
    def show_settings_dialog(self):
        """Show the Settings dialog."""
        from uexplorer_src.ui.dialogs.settings_dialog import SettingsDialog
        
        dialog = SettingsDialog(locator=self.locator, parent=self)
        if dialog.exec():
            logger.info("Settings saved")
        else:
            logger.info("Settings dialog cancelled")
    
    def new_window(self):
        """Open a new UExplorer window."""
        # TODO: Implement new window - would require app-level window management
        logger.info("New window requested (not yet implemented)")
    
    def new_browser(self):
        """
        Create a new file browser document.
        
        Spawns a new FilePaneWidget with its own ViewModel,
        adds it to DockingService as a closable document.
        """
        from uexplorer_src.ui.widgets.file_pane import FilePaneWidget
        import uuid
        
        # Generate unique ID
        browser_num = len([k for k in self.document_manager.documents if k.startswith("browser_")]) + 1
        doc_id = f"browser_{browser_num}"
        
        # Create FilePaneWidget
        pane = FilePaneWidget(self.locator)
        
        # Create and connect ViewModel
        if hasattr(self, 'document_manager'):
            viewmodel = self.document_manager.create_document(doc_id)
            pane.set_viewmodel(viewmodel)
        
        # Connect to managers
        if hasattr(self, 'filter_manager') and hasattr(self, 'selection_manager'):
            pane.set_managers(
                filter_manager=self.filter_manager,
                selection_manager=self.selection_manager
            )
            pane.find_similar.connect(self._on_find_similar_image)
        
        # Add to docking service
        if hasattr(self, 'docking_service'):
            self.docking_service.add_document(
                doc_id=doc_id,
                widget=pane,
                title=f"Browser {browser_num}",
                area="center",
                closable=True
            )
        
        # Set as active
        if hasattr(self, 'document_manager'):
            self.document_manager.set_active(doc_id)
        
        logger.info(f"✓ New browser created: {doc_id}")

