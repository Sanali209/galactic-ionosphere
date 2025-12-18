"""
Image Search Application - Main Window
"""
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from loguru import logger
import sys
from pathlib import Path

# Temporary path setup (until pip install -e foundation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "templates/foundation"))

from foundation import DockManager, ActionRegistry, MenuBuilder

class MainWindow(QMainWindow):
    """
    Main application window for Image Search.
    """
    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel
        self.setWindowTitle("Image Search - DuckDuckGo")
        self.resize(1400, 900)
        
        # Initialize GUI components
        self.action_registry = ActionRegistry(self)
        self.dock_manager = DockManager(self, viewmodel.locator.config)
        self.menu_builder = MenuBuilder(self, self.action_registry)
        
        # Setup UI
        self._setup_central_widget()
        self._register_panels()
        self._register_actions()
        self._build_menus()
        
        # Results storage
        self.current_results = []
        
        logger.info("Image Search MainWindow initialized")
        
        # Load any existing results from MongoDB
        import asyncio
        asyncio.create_task(self._load_latest_results())
    
    def _register_panels(self):
        """Register and create panels."""
        # Import SearchPanel using importlib
        import importlib.util
        import sys
        from pathlib import Path
        
        panel_path = Path(__file__).parent / "panels/search_panel.py"
        spec = importlib.util.spec_from_file_location("app_search_panel", panel_path)
        search_panel_mod = importlib.util.module_from_spec(spec)
        sys.modules["app_search_panel"] = search_panel_mod
        spec.loader.exec_module(search_panel_mod)
        SearchPanel = search_panel_mod.SearchPanel
        
        # Import GalleryPanel
        gallery_path = Path(__file__).parent / "panels/gallery_panel.py"
        spec = importlib.util.spec_from_file_location("app_gallery_panel", gallery_path)
        gallery_panel_mod = importlib.util.module_from_spec(spec)
        sys.modules["app_gallery_panel"] = gallery_panel_mod
        spec.loader.exec_module(gallery_panel_mod)
        GalleryPanel = gallery_panel_mod.GalleryPanel
        
        # Register panels
        self.dock_manager.register_panel("search", SearchPanel)
        self.dock_manager.register_panel("gallery", GalleryPanel)
        
        # Create and show search panel
        self.search_panel = self.dock_manager.create_panel("search")
        
        # Create gallery panel as central widget
        self.gallery_panel = GalleryPanel("Gallery", self.viewmodel.locator, self)
        self.setCentralWidget(self.gallery_panel)
        
        # Connect signals
        self.search_panel.search_requested.connect(self._execute_search)
        self.gallery_panel.selection_changed.connect(self._on_selection_changed)
    
    def _setup_central_widget(self):
        """Setup central area (gallery panel is set in _register_panels)."""        
        pass
    
    def _register_actions(self):
        """Register all application actions."""
        # File menu
        self.action_registry.register_action(
            "file_new_search", "&New Search", self._on_new_search, "Ctrl+N")
        self.action_registry.register_action(
            "file_open_downloads", "Open &Downloads Folder", self._on_open_downloads)
        self.action_registry.register_action(
            "file_exit", "E&xit", self.close, "Alt+F4")
        
        # Edit menu
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
    
    def _build_menus(self):
        """Build application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_registry.get_action("file_new_search"))
        file_menu.addAction(self.action_registry.get_action("file_open_downloads"))
        file_menu.addSeparator()
        file_menu.addAction(self.action_registry.get_action("file_exit"))
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_registry.get_action("edit_select_all"))
        edit_menu.addAction(self.action_registry.get_action("edit_deselect_all"))
        
        # Search menu
        search_menu = menubar.addMenu("&Search")
        search_menu.addAction(self.action_registry.get_action("search_images"))
        
        # Download menu
        download_menu = menubar.addMenu("&Download")
        download_menu.addAction(self.action_registry.get_action("download_selected"))
        download_menu.addAction(self.action_registry.get_action("download_all"))
    
    def _execute_search(self, query: str, count: int):
        """Execute search using TaskSystem."""
        logger.info(f"Executing search: '{query}' (count: {count})")
        
        import asyncio
        from foundation import TaskSystem
        
        # Update UI
        self.search_panel.set_searching(True)
        self.search_panel.add_to_history(query)
        self.statusBar().showMessage(f"Searching for '{query}'...")
        
        # Get task system (pass class,not string)
        task_system = self.viewmodel.locator.get_system(TaskSystem)
        
        # Submit task using handler pattern
        asyncio.create_task(self._run_search(query, count, task_system))
    
    async def _run_search(self, query: str, count: int, task_system):
        """Run search task and handle results."""
        try:
            # Import and register handler
            import importlib.util
            import sys
            from pathlib import Path
            import asyncio
            
            task_path = Path(__file__).parent.parent / "tasks/search_task.py"
            spec = importlib.util.spec_from_file_location("app_search_task", task_path)
            search_task_mod = importlib.util.module_from_spec(spec)
            sys.modules["app_search_task"] = search_task_mod
            spec.loader.exec_module(search_task_mod)
            
            # Register handler
            task_system.register_handler("search_images", search_task_mod.search_images_handler)
            
            # Submit task (args must be strings)
            await task_system.submit("search_images", f"Search: {query}", query, str(count))
            
            logger.info(f"Search task submitted successfully")
            
            # Wait a bit for task to complete
            await asyncio.sleep(2)
            
            # Load results from MongoDB
            await self._load_latest_results(query)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.statusBar().showMessage("Search failed")
        finally:
            # Re-enable UI
            self.search_panel.set_searching(False)
    
    async def _load_latest_results(self, query: str = None):
        """Load latest search results from MongoDB and display in gallery."""
        try:
            import app_image_record
            ImageRecord = app_image_record.ImageRecord
            
            # Query MongoDB for latest results
            if query:
                # Find by search history query
                import app_search_history
                SearchHistory = app_search_history.SearchHistory
                searches = await SearchHistory.find({"query": query}, sort=[("timestamp", -1)], limit=1)
                if searches:
                    search_id = searches[0]._id
                    images = await ImageRecord.find({"search_id": search_id})
                else:
                    images = []
            else:
                # Get all images (limit to 100)
                images = await ImageRecord.find({}, limit=100)
            
            logger.info(f"📊 Loaded {len(images)} results from MongoDB")
            
            # Display in gallery
            self.gallery_panel.load_images(images)
            
            # Update status
            if images:
                self.statusBar().showMessage(f"Showing {len(images)} images")
            else:
                self.statusBar().showMessage("No results found")
                
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            logger.exception("Traceback:")
    
    # Action handlers
    def _on_new_search(self):
        logger.info("New Search")
        self.search_panel.query_input.clear()
        self.search_panel.query_input.setFocus()
    
    def _on_open_downloads(self):
        logger.info("Open Downloads")
    
    def _on_select_all(self):
        logger.info("Select All")
        # TODO: Implement select all thumbnails
    
    def _on_deselect_all(self):
        logger.info("Deselect All")
        # TODO: Implement deselect all
    
    def _on_selection_changed(self, selected_records):
        """Handle gallery selection changes."""
        count = len(selected_records)
        if count > 0:
            self.statusBar().showMessage(f"{count} image(s) selected")
        else:
            total = len(self.gallery_panel.image_records)
            self.statusBar().showMessage(f"Showing {total} images")
    
    def _on_search(self):
        """Trigger search from menu."""
        self.search_panel.query_input.setFocus()
    
    def _on_download_selected(self):
        logger.info("Download Selected")
    
    def _on_download_all(self):
        logger.info("Download All")
