"""
Image Search Application - Main Window

Refactored to use Foundation MainWindow with CardView gallery.
"""
from pathlib import Path
from loguru import logger

# Foundation imports
from src.ui.main_window import MainWindow as FoundationMainWindow
from src.core.locator import ServiceLocator

# Local imports
from app_src.ui.panels.search_panel import SearchPanel
from app_src.ui.panels.gallery_panel import GalleryPanel


class MainWindow(FoundationMainWindow):
    """
    Main application window for Image Search.
    
    Extends Foundation MainWindow for:
    - DockingService panel management
    - ActionRegistry + MenuBuilder
    - Status bar updates
    """
    
    def __init__(self, viewmodel):
        """Initialize main window with viewmodel."""
        super().__init__(viewmodel)
        self.viewmodel = viewmodel
        self.setWindowTitle("Image Search - DuckDuckGo")
        self.resize(1400, 900)
        
        # Results storage
        self.current_results = []
        
        # Setup panels
        self._setup_panels()
        
        # Register app-specific actions
        self._register_app_actions()
        
        logger.info("Image Search MainWindow initialized")
        
        # Load any existing results
        import asyncio
        asyncio.create_task(self._load_latest_results())
    
    def _setup_panels(self):
        """Setup search and gallery panels."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget
        
        # Create search panel (docked left as standard QDockWidget)
        self.search_panel = SearchPanel("Search", self.viewmodel.locator, self)
        
        # Add as standard Qt dock widget for guaranteed visibility
        search_dock = QDockWidget("Search", self)
        search_dock.setWidget(self.search_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, search_dock)
        
        # Create gallery panel (central)
        self.gallery_panel = GalleryPanel("Gallery", self.viewmodel, self)
        self.setCentralWidget(self.gallery_panel)
        
        # Connect signals
        self.search_panel.search_requested.connect(self._execute_search)
        self.gallery_panel.selection_changed.connect(self._on_selection_changed)
    
    def _register_app_actions(self):
        """Register application-specific actions."""
        # File menu additions
        self.action_registry.register_action(
            "file_new_search", "&New Search", self._on_new_search, "Ctrl+N")
        self.action_registry.register_action(
            "file_open_downloads", "Open &Downloads Folder", self._on_open_downloads)
        
        # Edit menu additions
        self.action_registry.register_action(
            "edit_select_all", "Select &All", self._on_select_all, "Ctrl+A")
        self.action_registry.register_action(
            "edit_deselect_all", "&Deselect All", self._on_deselect_all, "Ctrl+D")
        
        # Search menu
        self.action_registry.register_action(
            "search_images", "&Search Images", self._on_search, "Ctrl+F")
        
        # Download menu
        self.action_registry.register_action(
            "download_selected", "Download &Selected", self._on_download_selected, "Ctrl+S")
        self.action_registry.register_action(
            "download_all", "Download &All", self._on_download_all, "Ctrl+Shift+S")
        
        # Build custom menus
        self._build_app_menus()
    
    def _build_app_menus(self):
        """Build application-specific menus."""
        menubar = self.menuBar()
        
        # Search menu
        search_menu = menubar.addMenu("&Search")
        search_menu.addAction(self.action_registry.get_action("search_images"))
        search_menu.addAction(self.action_registry.get_action("file_new_search"))
        
        # Download menu
        download_menu = menubar.addMenu("&Download")
        download_menu.addAction(self.action_registry.get_action("download_selected"))
        download_menu.addAction(self.action_registry.get_action("download_all"))
    
    # --- Search Execution ---
    
    def _execute_search(self, query: str, count: int):
        """Execute search using TaskSystem."""
        logger.info(f"Executing search: '{query}' (count: {count})")
        
        import asyncio
        from src.core.tasks.system import TaskSystem
        
        # Update UI
        self.search_panel.set_searching(True)
        self.search_panel.add_to_history(query)
        self.statusBar().showMessage(f"Searching for '{query}'...")
        
        # Get task system
        task_system = self.viewmodel.locator.get_system(TaskSystem)
        
        # Submit search task
        asyncio.create_task(self._run_search(query, count, task_system))
    
    async def _run_search(self, query: str, count: int, task_system):
        """Run search task and handle results."""
        try:
            from app_src.tasks.search_task import search_images_handler
            import asyncio
            
            # Register handler
            task_system.register_handler("search_images", search_images_handler)
            
            # Submit task
            await task_system.submit("search_images", f"Search: {query}", query, str(count))
            
            logger.info("Search task submitted successfully")
            
            # Wait for task to complete
            await asyncio.sleep(2)
            
            # Load results
            await self._load_latest_results(query)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.statusBar().showMessage("Search failed")
        finally:
            self.search_panel.set_searching(False)
    
    async def _load_latest_results(self, query: str = None):
        """Load latest search results and display in gallery."""
        try:
            from app_src.models.image_record import ImageRecord
            from app_src.models.search_history import SearchHistory
            
            if query:
                # Find by search history
                searches = await SearchHistory.find(
                    {"query": query}, 
                    sort=[("timestamp", -1)], 
                    limit=1
                )
                if searches:
                    search_id = searches[0]._id
                    images = await ImageRecord.find({"search_id": search_id})
                else:
                    images = []
            else:
                # Get all images (limit 100)
                images = await ImageRecord.find({}, limit=100)
            
            logger.info(f"📊 Loaded {len(images)} results from MongoDB")
            
            # Display in gallery
            await self.gallery_panel.load_images(images)
            
            # Update status
            if images:
                self.statusBar().showMessage(f"Showing {len(images)} images")
            else:
                self.statusBar().showMessage("No results found")
                
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
    
    # --- Action Handlers ---
    
    def _on_new_search(self):
        """Clear and focus search input."""
        logger.info("New Search")
        self.search_panel.query_input.clear()
        self.search_panel.query_input.setFocus()
    
    def _on_open_downloads(self):
        """Open downloads folder."""
        logger.info("Open Downloads")
        # TODO: Open downloads folder
    
    def _on_select_all(self):
        """Select all images in gallery."""
        logger.info("Select All")
        self.gallery_panel.select_all()
    
    def _on_deselect_all(self):
        """Deselect all images."""
        logger.info("Deselect All")
        self.gallery_panel.clear_selection()
    
    def _on_selection_changed(self, selected_items):
        """Handle gallery selection changes."""
        count = len(selected_items)
        if count > 0:
            self.statusBar().showMessage(f"{count} image(s) selected")
        else:
            total = len(self.gallery_panel.get_all_items())
            self.statusBar().showMessage(f"Showing {total} images")
    
    def _on_search(self):
        """Focus search input."""
        self.search_panel.query_input.setFocus()
    
    def _on_download_selected(self):
        """Download selected images."""
        logger.info("Download Selected")
        selected = self.gallery_panel.get_selected_items()
        logger.info(f"Downloading {len(selected)} images...")
        # TODO: Implement download
    
    def _on_download_all(self):
        """Download all images."""
        logger.info("Download All")
        all_items = self.gallery_panel.get_all_items()
        logger.info(f"Downloading {len(all_items)} images...")
        # TODO: Implement download
