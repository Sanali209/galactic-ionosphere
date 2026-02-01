"""
Panel Factory for UExplorer

Creates and registers all docking panels with DockingService.
Extracted from MainWindow for modularity.
"""
from typing import TYPE_CHECKING, Dict, Any, Callable
from loguru import logger

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from src.core.locator import ServiceLocator
    from src.ui.docking import DockingService


class PanelDefinition:
    """Definition for a panel to be created."""
    
    def __init__(
        self,
        panel_id: str,
        panel_class: type,
        title: str,
        area: str = "left",
        closable: bool = False,
        init_args: tuple = None,
        init_kwargs: dict = None,
    ):
        self.panel_id = panel_id
        self.panel_class = panel_class
        self.title = title
        self.area = area
        self.closable = closable
        self.init_args = init_args or ()
        self.init_kwargs = init_kwargs or {}


def create_all_panels(
    window: "QWidget",
    locator: "ServiceLocator",
    docking_service: "DockingService",
) -> Dict[str, Any]:
    """
    Create all docking panels for UExplorer.
    
    Args:
        window: Parent window for panels
        locator: ServiceLocator for accessing services
        docking_service: DockingService for panel registration
        
    Returns:
        Dict mapping panel_id to panel instance
    """
    panels = {}
    
    # Import panel classes
    from uexplorer_src.ui.docking.tag_panel import TagPanel
    from uexplorer_src.ui.docking.album_panel import AlbumPanel
    from uexplorer_src.ui.docking.properties_panel import PropertiesPanel
    from uexplorer_src.ui.docking.relations_panel import RelationsPanel
    from uexplorer_src.ui.docking.background_panel import BackgroundPanel
    from uexplorer_src.ui.docking.directory_panel import DirectoryPanel
    from uexplorer_src.ui.docking.unified_search_panel import UnifiedSearchPanel
    from uexplorer_src.ui.docking.similar_items_panel import SimilarItemsPanel
    from uexplorer_src.ui.docking.annotation_panel import AnnotationPanel
    from uexplorer_src.ui.docking.maintenance_panel import MaintenancePanel
    from uexplorer_src.ui.docking.context_monitor_panel import ContextMonitorPanel
    
    # === LEFT PANELS ===
    
    # Tags Panel
    panels["tags"] = TagPanel(window, locator)
    docking_service.add_panel(
        panel_id="tags",
        widget=panels["tags"],
        title="Tags",
        area="left",
        closable=False
    )
    
    # Albums Panel  
    panels["albums"] = AlbumPanel(window, locator)
    docking_service.add_panel(
        panel_id="albums",
        widget=panels["albums"],
        title="Albums",
        area="left",
        closable=False
    )
    
    # Directories Panel
    panels["directories"] = DirectoryPanel(window, locator)
    docking_service.add_panel(
        panel_id="directories",
        widget=panels["directories"],
        title="Directories",
        area="left",
        closable=False
    )
    
    # Unified Search Panel
    panels["search"] = UnifiedSearchPanel(locator=locator)
    docking_service.add_panel(
        panel_id="search",
        widget=panels["search"],
        title="Search",
        area="left",
        closable=True
    )
    
    # === RIGHT PANELS ===
    
    # Properties Panel
    panels["properties"] = PropertiesPanel(window, locator)
    docking_service.add_panel(
        panel_id="properties",
        widget=panels["properties"],
        title="Properties",
        area="right",
        closable=False
    )
    
    # === BOTTOM PANELS ===
    
    # Relations Panel
    panels["relations"] = RelationsPanel(window, locator)
    docking_service.add_panel(
        panel_id="relations",
        widget=panels["relations"],
        title="Related Files",
        area="bottom",
        closable=False
    )
    
    # Background Tasks Panel
    panels["background"] = BackgroundPanel(locator, window)
    docking_service.add_panel(
        panel_id="background",
        widget=panels["background"],
        title="Background Tasks",
        area="bottom",
        closable=True  # Can be closed to save performance (timer stops when hidden)
    )
    
    # Maintenance Panel
    panels["maintenance"] = MaintenancePanel()
    _setup_maintenance_panel(panels["maintenance"], locator)
    docking_service.add_panel(
        panel_id="maintenance",
        widget=panels["maintenance"],
        title="Maintenance",
        area="bottom",
        closable=True
    )
    
    # Similar Items Panel
    panels["similar"] = SimilarItemsPanel(window, locator)
    docking_service.add_panel(
        panel_id="similar",
        widget=panels["similar"],
        title="Similar Files",
        area="bottom",
        closable=False
    )
    
    # Annotation Panel
    panels["annotation"] = AnnotationPanel(window, locator)
    docking_service.add_panel(
        panel_id="annotation",
        widget=panels["annotation"],
        title="Annotation",
        area="bottom",
        closable=False
    )
    
    # Context Monitor (Diagnostics)
    panels["sync_monitor"] = ContextMonitorPanel(locator, window)
    docking_service.add_panel(
        panel_id="sync_monitor",
        widget=panels["sync_monitor"],
        title="Sync Monitor",
        area="bottom",
        closable=True
    )
    
    logger.info("✓ Created all tool panels")
    return panels


def _setup_maintenance_panel(panel, locator: "ServiceLocator"):
    """Initialize maintenance panel with services."""
    try:
        # Note: MaintenanceService is Engine-only and accessed via EngineProxy
        # The MaintenancePanel should use EngineProxy for all operations
        # Scheduler is still Client-side
        from src.core.scheduling import PeriodicTaskScheduler
        
        scheduler = locator.get_system(PeriodicTaskScheduler)
        
        if scheduler:
            # Pass only scheduler - panel will use EngineProxy for maintenance ops
            panel.set_services(None, scheduler)  # maintenance_service=None
            logger.info("MaintenancePanel connected to scheduler (uses EngineProxy for maintenance)")
    except Exception as e:
        logger.debug(f"MaintenancePanel init warning: {e}")


def connect_panel_signals(
    panels: Dict[str, Any],
    window: "QWidget",
    on_directory_selected: Callable = None,
    on_album_selected: Callable = None,
    on_relation_selected: Callable = None,
    on_search_requested: Callable = None,
    selection_manager = None,
):
    """
    Connect signals between panels and window.
    
    Args:
        panels: Dict of panel_id -> panel instance
        window: Parent window
        on_directory_selected: Handler for directory selection
        on_album_selected: Handler for album selection
        on_relation_selected: Handler for relation category selection
        on_search_requested: Handler for search requests
        selection_manager: SelectionManager instance
    """
    # Directory panel -> open browser
    if on_directory_selected and "directories" in panels:
        panels["directories"].directory_selected.connect(on_directory_selected)
    
    # Tag panel signals
    if "tags" in panels and hasattr(panels["tags"], 'tree'):
        panels["tags"].tree.files_dropped_on_tag.connect(
            lambda tag_id, files: logger.info(f"Tagged {len(files)} files with {tag_id}")
        )
    
    # Album panel -> filter
    if on_album_selected and "albums" in panels and hasattr(panels["albums"], 'tree'):
        panels["albums"].tree.album_selected.connect(on_album_selected)
    
    # Relations panel -> filter
    if on_relation_selected and "relations" in panels and hasattr(panels["relations"], 'tree'):
        panels["relations"].tree.category_selected.connect(on_relation_selected)
    
    # Similar items panel -> selection manager
    if selection_manager and "similar" in panels:
        panels["similar"].set_selection_manager(selection_manager)
    
    # Search panel -> search handler
    if on_search_requested and "search" in panels:
        panels["search"].search_requested.connect(on_search_requested)
        logger.info("Connected UnifiedSearchPanel.search_requested")
    
    logger.info("✓ Connected panel signals")
