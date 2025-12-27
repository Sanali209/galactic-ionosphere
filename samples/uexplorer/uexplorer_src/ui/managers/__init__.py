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

__all__ = [
    "FilterManager",
    "FilterState",
    "SelectionManager",
    "SelectionState",
    "MenuManager",
    "ToolbarManager",
]


