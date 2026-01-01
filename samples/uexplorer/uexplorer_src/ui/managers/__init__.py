"""
UExplorer UI Managers Package.

Provides centralized state management for the UI.

NOTE: UExplorer uses its own selection manager that QObject-based for direct
widget parenting. Foundation SelectionManager is available separately.
"""
from uexplorer_src.ui.managers.filter_manager import FilterManager, FilterState
from uexplorer_src.ui.managers.selection_manager import SelectionManager, SelectionState
from uexplorer_src.ui.managers.menu_manager import MenuManager
from uexplorer_src.ui.managers.toolbar_manager import ToolbarManager
from uexplorer_src.ui.managers.session_manager import (
    SessionManager,
    restore_browser_document,
    open_browser_for_directory,
)
from uexplorer_src.ui.managers.panel_factory import (
    create_all_panels,
    connect_panel_signals,
    PanelDefinition,
)

__all__ = [
    "FilterManager",
    "FilterState",
    "SelectionManager",
    "SelectionState",
    "MenuManager",
    "ToolbarManager",
    "SessionManager",
    "restore_browser_document",
    "open_browser_for_directory",
    "create_all_panels",
    "connect_panel_signals",
    "PanelDefinition",
]


