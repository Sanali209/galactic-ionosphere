"""
PySide6 UI Framework Bundle.

Optional bundle that registers Qt/PySide6-dependent UI services.
Only import this bundle when building a GUI application.

Usage:
    # GUI application
    builder = (ApplicationBuilder.for_gui("MyApp", "config.json")
        .add_bundle(PySideBundle(theme="dark"))
        .build())
    
    # Console application - don't use this bundle!
    builder = (ApplicationBuilder.for_console("CLI", "config.json")
        .build())  # No PySideBundle needed
"""
from typing import TYPE_CHECKING

from src.core.bootstrap import SystemBundle

if TYPE_CHECKING:
    from src.core.bootstrap import ApplicationBuilder


class PySideBundle(SystemBundle):
    """
    Bundle for PySide6 UI framework services.
    
    Registers:
    - DockingService: Window/panel docking system
    - SessionState: UI session persistence
    - NavigationService: Smart selection routing
    - DocumentManager: Document lifecycle management
    
    Requirements:
    - PySide6 must be installed
    - Only use for GUI applications
    
    Example:
        builder = (ApplicationBuilder.for_gui("UExplorer")
            .add_bundle(PySideBundle(theme="dark")))
    """
    
    def __init__(self, theme: str = "dark"):
        """
        Initialize PySideBundle.
        
        Args:
            theme: UI theme ("dark" or "light")
        """
        self.theme = theme
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """
        Register UI services (requires PySide6).
        
        Args:
            builder: ApplicationBuilder to register systems with
            
        Raises:
            RuntimeError: If PySide6 is not installed
        Register PySide6-dependent UI services.
        
        Note: DockingService is NOT registered here because it requires
        a parent QMainWindow which doesn't exist during builder.build().
        It should be initialized directly by MainWindow.__init__().
        """
        # Check PySide6 availability
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            logger.warning("PySide6 not available - UI bundle registration skipped")
            return
        
        # Register UI services that don't need parent widget
        from src.ui.state import SessionState
        from src.ui.navigation.service import NavigationService
        from src.ui.mvvm.sync_manager import ContextSyncManager
        
        builder.add_system(SessionState)        # UI session persistence
        builder.add_system(NavigationService)   # Smart Selection Routing
        builder.add_system(ContextSyncManager)   # Reactive Synchronization
        # Note: DocumentManager is registered per-application
        # It's not in this bundle because it might be Engine-only
        # Add it explicitly if needed:
        # from src.ui.mvvm.document_manager import DocumentManager
        # builder.add_system(DocumentManager)
        
        # Store theme preference (can be accessed by UI later)
        # This is optional metadata for the application
        if not hasattr(builder, '_metadata'):
            builder._metadata = {}
        builder._metadata['ui_theme'] = self.theme
