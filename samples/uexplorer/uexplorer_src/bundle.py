"""
UExplorer UI System Bundle.

Groups UExplorer-specific UI services for easy registration.
"""
from src.core.bootstrap import SystemBundle

if False:  # TYPE_CHECKING
    from src.core.bootstrap import ApplicationBuilder


class UExplorerUIBundle(SystemBundle):
    """
    Bundle for UExplorer UI-specific services.
    
    Includes:
    - SessionState: UI session persistence
    - NavigationService: Smart selection routing
    
    Example:
        builder = (ApplicationBuilder("UExplorer", "config.json")
                   .with_default_systems()
                   .add_bundle(UExplorerUIBundle())
                   .add_bundle(UCoreFSBundle()))
    """
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """Register UExplorer UI services."""
        from src.ui.state import SessionState
        from src.ui.navigation.service import NavigationService
        
        builder.add_system(SessionState)        # UI session persistence
        builder.add_system(NavigationService)   # Smart Selection Routing
