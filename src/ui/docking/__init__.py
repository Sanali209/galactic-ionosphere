"""
Docking system for panels and documents.

Uses PySide6-QtAds for professional docking experience.

Features:
- Document management (center area, tabs)
- Panel management (side areas, auto-hide)
- Perspectives (named layout configurations)
- Full state persistence support

Usage:
    from src.ui.docking import DockingService
    
    docking = DockingService(main_window)
    docking.add_document("editor", widget, "Editor")
    docking.add_panel("properties", props, "Properties", area="right")
"""
from .service import DockingService, DockArea, DockType, SideBar
from .panel_base import BasePanelWidget

__all__ = [
    "DockingService",
    "DockArea",
    "DockType", 
    "SideBar",
    "BasePanelWidget",
]
