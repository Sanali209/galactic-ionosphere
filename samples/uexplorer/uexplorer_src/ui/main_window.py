"""
UExplorer Main Window

Directory Opus-inspired dual-pane file manager.
Now powered by PySide6-QtAds for professional docking!
"""
from pathlib import Path
import asyncio

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QToolBar, QStatusBar, QLabel
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from loguru import logger

# Import DockingService from Foundation's docking module
from src.ui.docking import DockingService
from src.ui.navigation.service import NavigationService, NavigationContext

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
        
        # Connect Docking active state to DocumentManager
        if hasattr(self, 'docking_service') and hasattr(self, 'document_manager'):
             self.docking_service.focus_changed.connect(
                 self.document_manager.set_active
             )
        
        # Connect TaskSystem to UI progress
        self.setup_task_progress()
        
        # Apply theme
        self.apply_theme()
        
        
        # Connect SessionState to DockingService for integrated persistence
        self._setup_session_integration()
        
        # Restore session (documents + layout)
        # This acts as the single source of truth for startup state
        if not self._restore_session():
            logger.info("No session found - initializing default layout")
            # No session to restore - create default single browser
            self.new_browser()
            # Apply default layout if no saved layout exists
            self.restore_docking_layout()
        
        logger.info("UExplorer main window initialized")
    
    def _setup_session_integration(self):
        """Connect SessionState to DockingService for integrated persistence."""
        try:
            from src.ui.state import SessionState
            session = self.locator.get_system(SessionState)
            if session:
                session.set_docking_service(self.docking_service)
                logger.info("SessionState connected to DockingService")
        except (KeyError, ImportError) as e:
            logger.debug(f"SessionState not available: {e}")
    
    def _save_session(self):
        """Save session using Foundation's SessionState."""
        try:
            from src.ui.state import SessionState
            session = self.locator.get_system(SessionState)
            if session:
                session.save()
                logger.info("Session saved via SessionState")
                return
        except (KeyError, ImportError):
            pass
        
        # Fallback: save layout only
        try:
            self.save_layout()
        except RuntimeError:
            pass
    
    def _restore_session(self) -> bool:
        """Restore session using Foundation's SessionState. Returns True if documents restored."""
        try:
            from src.ui.state import SessionState
            session = self.locator.get_system(SessionState)
            if not session:
                return False
            
            # Get saved document states
            doc_states = session.get_document_states()
            if not doc_states:
                logger.info("No documents to restore from session")
                return False
            
            logger.info(f"Restoring session: {len(doc_states)} documents")
            restored = 0
            
            for doc_id, doc_state in doc_states.items():
                # Only restore documents that were actually open
                if doc_state.get("is_closed", False):
                    logger.debug(f"Skipping closed document: {doc_id}")
                    continue
                    
                try:
                    self._restore_browser_document(doc_id, doc_state)
                    restored += 1
                except Exception as e:
                    logger.error(f"Failed to restore {doc_id}: {e}")
            
            # Restore full layout from saved bytes (now works because objectNames match)
            docking_state = session.get("docking", {})
            layout = docking_state.get("layout_bytes") if isinstance(docking_state, dict) else None
            
            if layout and restored > 0:
                try:
                    layout_bytes = bytes.fromhex(layout)
                    self.docking_service.restore_layout(layout_bytes)
                    logger.info("Restored docking layout from session")
                except Exception as e:
                    logger.warning(f"Failed to restore layout: {e}")
                    # Fallback to panel visibility restoration
                    self._restore_panel_visibility(docking_state.get("panels", {}))
            else:
                # No layout bytes, try legacy or manual panel restoration
                panel_states = docking_state.get("panels", {}) if isinstance(docking_state, dict) else {}
                self._restore_panel_visibility(panel_states)
            
            logger.info(f"Session restored: {restored} documents")
            return restored > 0
            
        except (KeyError, ImportError) as e:
            logger.debug(f"SessionState not available: {e}")
            return False
    
    def _restore_browser_document(self, doc_id: str, doc_state: dict):
        """Restore a browser document from saved state."""
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        
        title = doc_state.get("title", "Files")
        custom_state = doc_state.get("custom_state", {})
        
        # Register with DocumentManager FIRST so it's tracked
        if hasattr(self, 'document_manager'):
            vm = self.document_manager.create_document(doc_id)
        else:
            vm = None
        
        # Pass viewmodel to document so search results work
        doc = FileBrowserDocument(self.locator, viewmodel=vm, title=title, parent=None)
        
        # Restore viewmodel state
        if custom_state:
            doc.set_state(custom_state)
        
        self.docking_service.add_document(doc_id, doc, title, area="center", closable=True)
        
        # Connect selection to metadata/properties panel
        doc.selection_changed.connect(self.on_selection_changed)
        
        logger.debug(f"Restored document: {doc_id} - {title}")
    
    def _restore_panel_visibility(self, panel_states: dict):
        """Restore panel visibility as fallback when layout_bytes fails."""
        if not panel_states:
            return
            
        for panel_id, panel_state in panel_states.items():
            try:
                is_visible = panel_state.get("is_visible", True)
                if is_visible:
                    self.docking_service.show_panel(panel_id)
                else:
                    self.docking_service.hide_panel(panel_id)
            except Exception as e:
                logger.debug(f"Failed to restore panel {panel_id}: {e}")
        logger.info("Restored panel visibility from session")
    
    def _open_browser_for_directory(self, directory_id: str):
        """Open a new browser tab and navigate to directory."""
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        import uuid
        
        doc_id = f"browser_{uuid.uuid4().hex[:8]}"
        
        # Create ViewModel via DocumentManager
        if hasattr(self, 'document_manager'):
            vm = self.document_manager.create_document(doc_id)
        else:
            vm = None
            
        doc = FileBrowserDocument(self.locator, viewmodel=vm, title="Files", parent=None)
        
        self.docking_service.add_document(doc_id, doc, "Files", area="center", closable=True)
        doc.browse_directory(directory_id)
    
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
        
        # --- Navigation System Setup ---
        # Get NavigationService
        try:
            self.navigation_service = self.locator.get_system(NavigationService)
            
            # Register DocumentManager as a handler for existing docs
            self.navigation_service.register_handler(self.document_manager)
            
            # Connect DocumentManager request to UI creation
            self.document_manager.request_new_document.connect(self._open_browser_for_directory)
            
            logger.info("NavigationService connected")
        except Exception as e:
            logger.error(f"Failed to setup NavigationService: {e}")
            
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
        """Handle window close event - save session via Foundation."""
        logger.info("=" * 50)
        logger.info("🚪 WINDOW CLOSE - Saving session")
        logger.info("=" * 50)
        
        # Save complete session (documents, layout, panels)
        self._save_session()
        
        # Accept the close event
        event.accept()
        logger.info("Window closed")
    
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
        """Create menu bar using MenuManager."""
        from uexplorer_src.ui.managers import MenuManager
        
        self.menu_manager = MenuManager(self, self.action_registry)
        self.menu_manager.build_menus()
        
        logger.info("Menu bar created via MenuManager")
    
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
        panels_menu.addAction(self.action_registry.get_action("view.panel.background"))
        
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
        """Create toolbar using ToolbarManager."""
        from uexplorer_src.ui.managers import ToolbarManager
        
        self.toolbar_manager = ToolbarManager(self, self.action_registry)
        self.toolbar = self.toolbar_manager.build_toolbar()
        
        logger.info("Toolbar created via ToolbarManager")
    
    def create_central_widget(self):
        """Create the central widget area with CardView file browser."""
        from src.ui.documents.split_manager import SplitManager
        from src.ui.documents.split_container import SplitContainer
        
        self.split_manager = SplitManager()
        
        # Create root container (empty initially)
        # Documents will be added by DockingService during session restore
        root_container = SplitContainer(self.split_manager.root.id)
        
        # Store container widget reference
        self.split_manager.set_container_widget(self.split_manager.root.id, root_container)
        
        # Set as central widget
        self.setCentralWidget(root_container)
        
        logger.info("✓ Central widget setup complete with drag & drop enabled")
    
    def create_left_panel(self):
        """Create left navigation panel."""
        from PySide6.QtWidgets import QTabWidget, QTreeWidget, QTreeWidgetItem
        import sys
        from pathlib import Path
        
        # Add widgets to path for imports
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
        from pathlib import Path
        
        # Add documents to path
        # Central widget - File browser with Foundation's SplitManager (drag & drop enabled)
        from pathlib import Path
        
        # Add Foundation to path
        from src.ui.documents.split_manager import SplitManager, SplitOrientation
        from src.ui.documents.split_container import SplitContainer
        
        # Import local file_pane_document
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
        from pathlib import Path
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
    
    def new_window(self):
        """Open a new UExplorer window."""
        # TODO: Implement multi-window support
        self.status_label.setText("Multi-window not yet implemented")
        logger.info("New window requested")
    
    def new_browser(self):
        """Create a new CardView file browser document tab."""
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        
        # Generate unique doc_id
        doc_count = len([k for k in self.docking_service._documents if k.startswith("browser_")])
        doc_id = f"browser_{doc_count + 1}"
        
        # Create ViewModel via DocumentManager
        vm = None
        if hasattr(self, 'document_manager'):
            vm = self.document_manager.create_document(doc_id)
        
        # Create FileBrowserDocument
        doc = FileBrowserDocument(
            locator=self.locator,
            viewmodel=vm,
            title=f"Browser {doc_count + 1}"
        )
        
        # Add to DockingService
        self.docking_service.add_document(
            doc_id=doc_id,
            widget=doc,
            title=f"Browser {doc_count + 1}",
            area="center",
            closable=True
        )
        
        # Connect selection signal
        doc.selection_changed.connect(self.on_selection_changed)
        
        # Activate new document
        self.docking_service.activate_document(doc_id)
        
        self.status_label.setText(f"Opened Browser {doc_count + 1}")
        logger.info(f"Created new browser: {doc_id}")
    
    def show_library_dialog(self):
        """Show library settings dialog."""
        from pathlib import Path
        
        # Import from correct location
        from uexplorer_src.ui.dialogs.library_dialog import LibraryDialog
        dialog = LibraryDialog(self.locator, parent=self)
        dialog.exec()
        
        # Refresh all active file browsers
        import asyncio
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        
        for doc in self.docking_service.documents.values():
            if isinstance(doc, FileBrowserDocument) and doc.view_model:
                asyncio.ensure_future(doc.view_model.refresh())
    
    def reprocess_selection(self):
        """Reprocess selected files through Phase 2/3 pipeline."""
        import asyncio
        from bson import ObjectId
        
        # Get selected file IDs from SelectionManager
        if not hasattr(self, 'selection_manager'):
            logger.warning("SelectionManager not available")
            return
        
        selected_ids = self.selection_manager.get_selected_ids()
        if not selected_ids:
            self.status_label.setText("No files selected to reprocess")
            return
        
        # Convert to ObjectIds
        object_ids = [ObjectId(str(fid)) for fid in selected_ids]
        
        async def _enqueue():
            try:
                from src.ucorefs.processing.pipeline import ProcessingPipeline
                pipeline = self.locator.get_system(ProcessingPipeline)
                
                if not pipeline:
                    self.status_label.setText("ProcessingPipeline not available")
                    return
                
                # Enqueue Phase 2 (metadata, thumbnails, embeddings)
                task_id = await pipeline.enqueue_phase2(object_ids, force=True)
                
                if task_id:
                    self.status_label.setText(f"Queued {len(object_ids)} files for reprocessing")
                    logger.info(f"Reprocess: Queued {len(object_ids)} files - Task {task_id}")
                else:
                    self.status_label.setText("Files already in processing queue")
                
            except Exception as e:
                logger.error(f"Failed to queue reprocessing: {e}")
                self.status_label.setText(f"Reprocess failed: {e}")
        
        # Run async
        asyncio.ensure_future(_enqueue())
    
    def reindex_all_files(self):
        """Reindex all unprocessed files in database via background tasks."""
        import asyncio
        
        async def _reindex():
            try:
                from src.ucorefs.processing.pipeline import ProcessingPipeline
                pipeline = self.locator.get_system(ProcessingPipeline)
                
                if not pipeline:
                    self.status_label.setText("ProcessingPipeline not available")
                    return
                
                self.status_label.setText("Starting reindex...")
                
                # Reindex unprocessed files
                result = await pipeline.reindex_all(include_processed=False)
                
                total = result.get("total_files", 0)
                batches = result.get("batches_queued", 0)
                
                if total > 0:
                    self.status_label.setText(f"Reindex: {total} files in {batches} batches queued")
                    logger.info(f"Reindex started: {total} files, {batches} tasks")
                else:
                    self.status_label.setText("No unprocessed files to reindex")
                
            except Exception as e:
                logger.error(f"Failed to reindex: {e}")
                self.status_label.setText(f"Reindex failed: {e}")
        
        asyncio.ensure_future(_reindex())
    
    # ==================== Maintenance Menu Actions ====================
    
    def rebuild_all_counts(self):
        """Show progress dialog and rebuild all counts."""
        asyncio.ensure_future(self._rebuild_counts_async())
    
    async def _rebuild_counts_async(self):
        """Execute count rebuild with progress feedback."""
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        from src.ucorefs.services.maintenance_service import MaintenanceService
        
        try:
            maintenance = self.locator.get_system(MaintenanceService)
            if not maintenance:
                QMessageBox.warning(self, "Error", "MaintenanceService not available")
                return
            
            # Show progress dialog
            progress = QProgressDialog("Rebuilding file counts...", "Cancel", 0, 3, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            try:
                progress.setValue(1)
                progress.setLabelText("Recalculating file counts across all systems...")
                result = await maintenance.rebuild_all_counts()
                
                if progress.wasCanceled():
                    return
                
                progress.setValue(3)
                
                # Show results
                total_updated = (result.get("tags_updated", 0) + 
                               result.get("albums_updated", 0) + 
                               result.get("directories_updated", 0))
                
                message = (
                    f"Count rebuild complete!\n\n"
                    f"Tags updated: {result.get('tags_updated', 0)}\n"
                    f"Albums updated: {result.get('albums_updated', 0)}\n"
                    f"Directories updated: {result.get('directories_updated', 0)}\n"
                    f"Duration: {result.get('duration', 0):.2f}s"
                )
                
                if result.get("errors"):
                    message += f"\n\nErrors: {len(result['errors'])}"
                
                QMessageBox.information(self, "Rebuild Complete", message)
                
                # Refresh all panels
                if hasattr(self, 'tags_panel') and self.tags_panel:
                    asyncio.ensure_future(self.tags_panel._tree.refresh_tags())
                if hasattr(self, 'albums_panel') and self.albums_panel:
                    asyncio.ensure_future(self.albums_panel._tree.refresh_albums())
                if hasattr(self, 'directory_panel') and self.directory_panel:
                    asyncio.ensure_future(self.directory_panel.on_update())
                
                logger.info(f"Count rebuild complete: {total_updated} records updated")
                
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Rebuild failed: {e}")
            logger.error(f"Count rebuild failed: {e}")
    
    def verify_references(self):
        """Verify data integrity."""
        asyncio.ensure_future(self._verify_references_async())
    
    async def _verify_references_async(self):
        """Verify ObjectId references with progress feedback."""
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        from src.ucorefs.services.maintenance_service import MaintenanceService
        
        try:
            maintenance = self.locator.get_system(MaintenanceService)
            if not maintenance:
                QMessageBox.warning(self, "Error", "MaintenanceService not available")
                return
            
            # Show progress dialog
            progress = QProgressDialog("Verifying data integrity...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            try:
                result = await maintenance.verify_references()
                
                if progress.wasCanceled():
                    return
                
                # Show results
                total_broken = (result.get("broken_tag_refs", 0) + 
                              result.get("broken_album_refs", 0) + 
                              result.get("broken_dir_refs", 0))
                
                if total_broken == 0:
                    message = f"All references are valid!\n\nFiles checked: {result.get('files_checked', 0)}"
                    QMessageBox.information(self, "Verification Complete", message)
                else:
                    message = (
                        f"Found broken references:\n\n"
                        f"Broken tag references: {result.get('broken_tag_refs', 0)}\n"
                        f"Broken album references: {result.get('broken_album_refs', 0)}\n"
                        f"Broken directory references: {result.get('broken_dir_refs', 0)}\n"
                        f"Files checked: {result.get('files_checked', 0)}\n\n"
                        f"Run 'Cleanup Orphaned Records' to fix."
                    )
                    QMessageBox.warning(self, "Broken References Found", message)
                
                logger.info(f"Reference verification complete: {total_broken} broken references")
                
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Verification failed: {e}")
            logger.error(f"Reference verification failed: {e}")
    
    def cleanup_orphaned_records(self):
        """Cleanup orphaned references."""
        asyncio.ensure_future(self._cleanup_orphaned_async())
    
    async def _cleanup_orphaned_async(self):
        """Cleanup orphaned records with confirmation."""
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        from src.ucorefs.services.maintenance_service import MaintenanceService
        
        try:
            # Confirm action
            reply = QMessageBox.question(
                self,
                "Cleanup Orphaned Records",
                "This will remove invalid references from your database.\n\n"
                "This operation is safe but cannot be undone.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            maintenance = self.locator.get_system(MaintenanceService)
            if not maintenance:
                QMessageBox.warning(self, "Error", "MaintenanceService not available")
                return
            
            # Show progress dialog
            progress = QProgressDialog("Cleaning up orphaned records...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            try:
                result = await maintenance.cleanup_orphaned_records()
                
                if progress.wasCanceled():
                    return
                
                # Show results
                message = (
                    f"Cleanup complete!\n\n"
                    f"Files cleaned: {result.get('files_cleaned', 0)}\n"
                    f"Tag references removed: {result.get('tags_removed', 0)}\n"
                    f"Album references removed: {result.get('albums_removed', 0)}"
                )
                
                if result.get("errors"):
                    message += f"\n\nErrors: {len(result['errors'])}"
                
                QMessageBox.information(self, "Cleanup Complete", message)
                logger.info(f"Cleanup complete: {result.get('files_cleaned', 0)} files cleaned")
                
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cleanup failed: {e}")
            logger.error(f"Cleanup failed: {e}")
    
    # ==================== End Maintenance Actions ====================
    
    def show_settings_dialog(self):

        """Show settings dialog integrated with ConfigManager."""
        try:
            from uexplorer_src.ui.dialogs.settings_dialog import SettingsDialog
            
            # ConfigManager is available via locator.config
            config = self.locator.config
            dialog = SettingsDialog(config_manager=config, locator=self.locator, parent=self)
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
    
    def _on_directory_selected(self, directory_id: str, path: str):
        """
        Handle directory selection from DirectoryPanel.
        
        Opens a FileBrowserDocument tab with files from the selected folder.
        
        Args:
            directory_id: MongoDB ObjectId of directory
            path: Filesystem path of directory
        """
        logger.info(f"Directory selected: {path} (ID: {directory_id})")
        
        # Use NavigationService for intelligent routing
        if hasattr(self, 'navigation_service'):
            handled = self.navigation_service.navigate(
                directory_id,  # Pass ID as data
                context=NavigationContext(
                    source_id="sidebar_directory_panel",
                    metadata={"path": path}
                )
            )
            if handled:
                return
        
        # Fallback if service not available (should not happen)
        logger.warning("NavigationService failed/missing, using fallback")
        self._open_browser_for_directory(directory_id)

    
    
    def setup_docking_service(self):
        """Initialize NEW docking system with DockingService (PySide6-QtAds)."""
        logger.info("Setting up DockingService (PySide6-QtAds)...")
        
        # Create docking service
        self.docking_service = DockingService(self)
        
        # Connect document activation to DocumentManager
        self.docking_service.document_activated.connect(self._on_document_tab_activated)
        
        # Register in locator for global access
        self.locator.register_instance(DockingService, self.docking_service)
        
        # Create tool panels (sides)
        # Documents will be added later by session restore or new_browser()
        self._create_tool_panels()
        
        logger.info("✓ DockingService initialized successfully!")
    
    def _on_document_tab_activated(self, doc_id: str):
        """Handle document tab activation - update DocumentManager."""
        if hasattr(self, 'document_manager'):
            self.document_manager.set_active(doc_id)
            logger.info(f"Active document switched to: {doc_id}")
    
    def _create_file_documents(self):
        """Create CardView-based file browser documents."""
        import json
        from pathlib import Path
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        
        # Restore session
        session_browsers = []
        try:
            from pathlib import Path
            import json
            session_file = Path(__file__).parent.parent.parent / "session.json"
            if session_file.exists():
                session_data = json.loads(session_file.read_text())
                
                # Check for new format (list of states)
                if "browsers" in session_data:
                    session_browsers = session_data["browsers"]
                    logger.debug(f"Session restore: {len(session_browsers)} browsers with state")
                else:
                    # Legacy fallback (count only)
                    count = min(session_data.get("browser_count", 1), 5)
                    # Generate placeholder states
                    for i in range(1, count + 1):
                        session_browsers.append({
                            "id": f"browser_{i}",
                            "title": f"Browser {i}"
                        })
                    logger.debug(f"Session restore: {count} browsers (legacy)")
                    
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            # Default to 1 browser
            session_browsers = [{"id": "browser_1", "title": "Files"}]
        
        # Ensure at least one browser if empty list passed
        if not session_browsers:
            # Check if this is a fresh start (no session file)
            # If so, return empty so new_browser() is called in __init__
            return

        # Create CardView-based browsers from state
        for i, state in enumerate(session_browsers):
            # Use ID from state or generate new
            doc_id = state.get("id", f"browser_{i+1}")
            title = state.get("title", f"Browser {i+1}")
            
            # Create ViewModel via DocumentManager first
            if hasattr(self, 'document_manager'):
                vm = self.document_manager.create_document(doc_id)
            else:
                vm = None
            
            # Create FileBrowserDocument with ViewModel
            doc = FileBrowserDocument(
                locator=self.locator,
                viewmodel=vm,
                title=title
            )
            
            # Restore internal state (directory, view mode, etc)
            doc.set_state(state)
            
            # Add to DockingService
            self.docking_service.add_document(
                doc_id=doc_id,
                widget=doc,
                title=title,
                area="center",
                closable=True
            )
            
            # Connect selection to metadata panel
            doc.selection_changed.connect(self.on_selection_changed)
            
            # Trigger directory load if restored
            if vm and vm.current_directory:
                 # Small delay to let UI settle
                 import asyncio
                 asyncio.create_task(self._delayed_browse(doc, str(vm.current_directory)))
            
            # Keep first for compatibility
            if i == 0:
                self.file_pane_left = doc
        
        logger.info(f"✓ Created {len(session_browsers)} CardView browser(s)")

    async def _delayed_browse(self, doc, dir_id):
        """Helper to trigger browse after UI init."""
        await asyncio.sleep(0.1)
        doc.browse_directory(dir_id)
    
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
        from pathlib import Path
        from uexplorer_src.ui.docking.tag_panel import TagPanel
        from uexplorer_src.ui.docking.album_panel import AlbumPanel
        from uexplorer_src.ui.docking.properties_panel import PropertiesPanel
        from uexplorer_src.ui.docking.relations_panel import RelationsPanel
        from uexplorer_src.ui.docking.background_panel import BackgroundPanel
        from uexplorer_src.ui.docking.directory_panel import DirectoryPanel
        
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
        
        # DIRECTORIES PANEL (Left, below albums)
        self.directories_panel = DirectoryPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="directories",
            widget=self.directories_panel,
            title="Directories",
            area="left",
            closable=False
        )
        # Connect directory selection to open file browser document
        self.directories_panel.directory_selected.connect(self._on_directory_selected)
        
        # UNIFIED SEARCH PANEL (Left, combines Search + Filter)
        from uexplorer_src.ui.docking.unified_search_panel import UnifiedSearchPanel
        self.unified_search_panel = UnifiedSearchPanel(locator=self.locator)
        self.docking_service.add_panel(
            panel_id="search",
            widget=self.unified_search_panel,
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
        
        # BACKGROUND PANEL (Bottom)
        self.background_panel = BackgroundPanel(self.locator, self)
        self.docking_service.add_panel(
            panel_id="background",
            widget=self.background_panel,
            title="Background Tasks",
            area="bottom",
            closable=False
        )
        
        # SIMILAR ITEMS PANEL (Bottom, next to relations)
        from uexplorer_src.ui.docking.similar_items_panel import SimilarItemsPanel
        self.similar_items_panel = SimilarItemsPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="similar",
            widget=self.similar_items_panel,
            title="Similar Files",
            area="bottom",
            closable=False
        )
        # Connect to SelectionManager
        if hasattr(self, 'selection_manager'):
            self.similar_items_panel.set_selection_manager(self.selection_manager)
        
        # ANNOTATION PANEL (Bottom)
        from uexplorer_src.ui.docking.annotation_panel import AnnotationPanel
        self.annotation_panel = AnnotationPanel(self, self.locator)
        self.docking_service.add_panel(
            panel_id="annotation",
            widget=self.annotation_panel,
            title="Annotation",
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
        
        # Connect UnifiedSearchPanel to execute search via pipeline
        self.unified_search_panel.search_requested.connect(self._on_search_requested)
        logger.info("Connected UnifiedSearchPanel.search_requested to _on_search_requested")
        
        # Setup UnifiedQueryBuilder to collect filters from all panels
        self._setup_unified_query_builder()
        
        logger.info("✓ Created all tool panels")
    
    def _setup_unified_query_builder(self):
        """Initialize and connect UnifiedQueryBuilder to all filter panels."""
        from uexplorer_src.viewmodels.unified_query_builder import UnifiedQueryBuilder
        
        self.query_builder = UnifiedQueryBuilder(self.locator, self)
        
        # Connect all panels
        self.query_builder.connect_all_panels(
            search_panel=getattr(self, 'unified_search_panel', None),
            tag_panel=getattr(self, 'tags_panel', None),
            album_panel=getattr(self, 'albums_panel', None),
            directory_panel=getattr(self, 'directories_panel', None)
        )
        
        # Connect UnifiedSearchPanel to query builder
        if hasattr(self, 'unified_search_panel'):
            self.unified_search_panel.set_query_builder(self.query_builder)
        
        # Connect to execute unified search when filters change
        self.query_builder.query_changed.connect(self._on_unified_query_changed)
        
        logger.info("✓ UnifiedQueryBuilder connected to all panels")
    
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
    
    def _on_unified_query_changed(self, unified_query):
        """
        Handle unified query change from any panel.
        
        This is the primary entry point for the new unified search system.
        Called when search panel, tag panel, album panel, or filter panel changes.
        """
        import asyncio
        asyncio.ensure_future(self._execute_unified_search(unified_query))
    
    async def _execute_unified_search(self, unified_query):
        """Execute search with unified query from all panels."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        
        logger.info(f"Unified search: mode={unified_query.mode}, "
                   f"text='{unified_query.text[:30]}...', "
                   f"filters={unified_query.has_filters()}")
        
        # Build SearchQuery from UnifiedSearchQuery
        search_query = SearchQuery(
            text=unified_query.text,
            mode=unified_query.mode,
            fields=unified_query.text_fields if unified_query.text_fields else ["name"],
            limit=unified_query.limit,
            # Pass through include/exclude filters
            filters={
                "tag_include": unified_query.tag_include,
                "tag_exclude": unified_query.tag_exclude,
                "album_include": unified_query.album_include,
                "album_exclude": unified_query.album_exclude,
                "directory_include": unified_query.directory_include,
                "directory_exclude": unified_query.directory_exclude,
                **unified_query.filters
            }
        )
        
        # For similar mode, set file_id
        if unified_query.mode == "similar" and unified_query.similar_file_id:
            search_query.file_id = unified_query.similar_file_id
        
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
            
            # Save browser states for session restore
            if hasattr(self, 'docking_service'):
                import json
                session_file = Path(__file__).parent.parent.parent / "session.json"
                
                browser_states = []
                # Iterate all documents in docking service
                for doc_id, dock_widget in self.docking_service.documents.items():
                    # Get actual widget from CDockWidget
                    widget = dock_widget.widget()
                    if hasattr(widget, 'get_state'):
                        state = widget.get_state()
                        state['id'] = doc_id  # Ensure ID is preserved
                        browser_states.append(state)
                
                session_data = {
                    "browser_count": len(browser_states),
                    "browsers": browser_states
                }
                session_file.write_text(json.dumps(session_data, default=str)) # default=str for ObjectId
                logger.debug(f"Session saved: {len(browser_states)} browsers with state")
            
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
        
        from pathlib import Path
        import uuid
        
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
                    
                    # Generate unique ID for new browser
                    idx = 2
                    new_doc_id = f"browser_{idx}"
                    while self.document_manager.documents and new_doc_id in self.document_manager.documents:
                        idx += 1
                        new_doc_id = f"browser_{idx}"
                    
                    title = f"Browser {idx}"
                    
                    # Create ViewModel via DocumentManager
                    if hasattr(self, 'document_manager'):
                        vm = self.document_manager.create_document(new_doc_id)
                    else:
                        vm = None
                        
                    # Create new FileBrowserDocument
                    from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
                    new_pane = FileBrowserDocument(self.locator, vm, title)
                    
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
                    
                    logger.info(f"Split horizontal - {title} created and registered")
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
        
        from pathlib import Path
        from pathlib import Path
        
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
                    
                    # Generate unique ID for new browser
                    idx = 2
                    new_doc_id = f"browser_{idx}"
                    while self.document_manager.documents and new_doc_id in self.document_manager.documents:
                        idx += 1
                        new_doc_id = f"browser_{idx}"
                    
                    title = f"Browser {idx}"
                    
                    # Create ViewModel via DocumentManager
                    if hasattr(self, 'document_manager'):
                        vm = self.document_manager.create_document(new_doc_id)
                    else:
                        vm = None
                        
                    # Create new FileBrowserDocument
                    from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
                    new_pane = FileBrowserDocument(self.locator, title, vm)
                    
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
                    
                    logger.info(f"Split vertical - {title} created and registered")
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

    def new_browser(self):
        """Open a new file browser tab."""
        import uuid
        from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
        
        # Determine ID
        doc_id = f"browser_{uuid.uuid4().hex[:8]}"
        
        # Create ViewModel via DocumentManager
        if hasattr(self, 'document_manager'):
            vm = self.document_manager.create_document(doc_id)
        else:
            vm = None
            
        doc = FileBrowserDocument(self.locator, viewmodel=vm, title="Files", parent=None)
        
        self.docking_service.add_document(doc_id, doc, "Files", area="center", closable=True)
        logger.info(f"New browser created: {doc_id}")
    
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
        """
        Reset layout to default configuration:
        Left: Directories
        Center: Files (Browser 1)
        Right: Properties
        Hidden: All others
        """
        from loguru import logger
        from pathlib import Path
        
        try:
            # 1. Clear saved persistence files
            layout_file = Path(__file__).parent.parent.parent / "docking_layout.bin"
            session_file = Path(__file__).parent.parent.parent / "session.json"
            
            if layout_file.exists():
                layout_file.unlink()
            if session_file.exists():
                session_file.unlink()
            
            logger.info("Cleared persistence files")
            
            # 2. Reset central area (Close all except Browser 1)
            # We must iterate a copy of keys since we modify the dict
            doc_ids = list(self.docking_service.documents.keys())
            for doc_id in doc_ids:
                if doc_id != "browser_1":
                    try:
                        self.docking_service.close_document(doc_id)
                    except RuntimeError:
                        pass # Valid C++ object check handled inside service, but double check here
            
            # Ensure "browser_1" exists
            # We need to process events to ensure closures happen before we check/add
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            if "browser_1" not in self.docking_service.documents:
               logger.info("Creating default browser_1")
               # We need to manually inject browser_1 if it was lost
               # But simpler to just call new_browser and rename/track it if needed
               # For now, relying on new_browser logic which auto-generates IDs might be tricky if we want EXACTLY "browser_1"
               # Let's just create a new one.
               self.new_browser() 
            
            # 3. Configure Panels (Programmatic Docking)
            # Define default visibility
            defaults = {
                "directories": True,
                "properties": True,
                "tags": False,
                "albums": False,
                "relations": False,
                "filters": False,
                "search": False,
                "background": False,
                "similar": False,
                "annotation": False
            }
            
            for panel_id, make_visible in defaults.items():
                if make_visible:
                    try:
                        self.docking_service.show_panel(panel_id)
                    except Exception: pass
                else:
                    try:
                        self.docking_service.hide_panel(panel_id)
                    except Exception: pass
                    
            logger.info("Layout reset to defaults")
            
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
