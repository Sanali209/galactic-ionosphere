"""
UExplorer Action Definitions

Centralized registry of all application actions using Foundation's ActionRegistry.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.menus.action_registry import ActionRegistry
    from ..main_window import MainWindow


def register_all_actions(registry: 'ActionRegistry', window: 'MainWindow'):
    """
    Register all UExplorer actions with the ActionRegistry.
    
    Args:
        registry: Foundation ActionRegistry instance
        window: MainWindow instance for callbacks
    """
    
    # ============================================================
    # FILE MENU
    # ============================================================
    
    registry.register_action(
        "file.new_window",
        "&New Window",
        window.new_window,
        shortcut="Ctrl+N",
        tooltip="Open a new UExplorer window"
    )
    
    registry.register_action(
        "file.new_browser",
        "New &Browser",
        window.new_browser,
        shortcut="Ctrl+T",
        tooltip="Open a new file browser panel"
    )
    
    registry.register_action(
        "file.exit",
        "E&xit",
        window.close,
        shortcut="Ctrl+Q",
        tooltip="Exit UExplorer"
    )
    
    # ============================================================
    # EDIT MENU
    # ============================================================
    
    registry.register_action(
        "edit.settings",
        "&Settings...",
        window.show_settings_dialog,
        shortcut="Ctrl+,",
        tooltip="Open application settings"
    )
    
    # ============================================================
    # VIEW MENU - PANELS
    # ============================================================
    
    registry.register_action(
        "view.panel.tags",
        "&Tags",
        lambda: window._toggle_panel("tags"),
        shortcut="Ctrl+1",
        tooltip="Toggle Tags panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.albums",
        "&Albums",
        lambda: window._toggle_panel("albums"),
        shortcut="Ctrl+2",
        tooltip="Toggle Albums panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.relations",
        "&Relations",
        lambda: window._toggle_panel("relations"),
        shortcut="Ctrl+3",
        tooltip="Toggle Relations panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.properties",
        "&Properties",
        lambda: window._toggle_panel("properties"),
        shortcut="Ctrl+4",
        tooltip="Toggle Properties panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.filters",
        "&Filters",
        lambda: window._toggle_panel("filters"),
        shortcut="Ctrl+5",
        tooltip="Toggle Filters panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.search",
        "&Search",
        lambda: window._toggle_panel("search"),
        shortcut="Ctrl+6",
        tooltip="Toggle Search panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.background",
        "&Background Tasks",
        lambda: window._toggle_panel("background"),
        shortcut="Ctrl+7",
        tooltip="Toggle Background Tasks panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.directories",
        "&Directories",
        lambda: window._toggle_panel("directories"),
        shortcut="Ctrl+8",
        tooltip="Toggle Directories panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.similar",
        "&Similar Files",
        lambda: window._toggle_panel("similar"),
        shortcut="Ctrl+9",
        tooltip="Toggle Similar Files panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.annotation",
        "&Annotation",
        lambda: window._toggle_panel("annotation"),
        shortcut="Ctrl+0",
        tooltip="Toggle Annotation panel",
        checkable=True
    )
    
    registry.register_action(
        "view.panel.maintenance",
        "&Maintenance",
        lambda: window._toggle_panel("maintenance"),
        shortcut="Ctrl+Shift+M",
        tooltip="Toggle Maintenance panel",
        checkable=True
    )
    
    # ============================================================
    # VIEW MENU - SPLITS
    # ============================================================

    
    registry.register_action(
        "view.dashboard",
        "&Dashboard",
        window.open_dashboard,
        shortcut="Ctrl+Home",
        tooltip="Open System Dashboard"
    )

    registry.register_action(
        "view.split_horizontal",
        "Split &Horizontal",
        window._split_horizontal,
        shortcut="Ctrl+Shift+H",
        tooltip="Split current pane horizontally (side-by-side)"
    )
    
    registry.register_action(
        "view.split_vertical",
        "Split &Vertical",
        window._split_vertical,
        shortcut="Ctrl+Shift+V",
        tooltip="Split current pane vertically (top-bottom)"
    )
    
    registry.register_action(
        "view.close_split",
        "&Close Split",
        window._close_split,
        shortcut="Ctrl+Shift+W",
        tooltip="Close current split and merge with sibling"
    )
    
    registry.register_action(
        "view.reset_layout",
        "&Reset Layout",
        window.reset_layout,
        tooltip="Reset all panels and splits to default layout"
    )
    
    # ============================================================
    # VIEW MENU - THUMBNAILS
    # ============================================================
    
    registry.register_action(
        "view.thumbnails",
        "Show &Thumbnails",
        lambda: None,  # TODO: Implement thumbnail toggle
        checkable=True,
        tooltip="Show/hide thumbnails in file view"
    )
    
    # ============================================================
    # TOOLS MENU
    # ============================================================
    
    registry.register_action(
        "tools.scan",
        "&Scan Directories",
        lambda: None,  # TODO: Implement scan trigger
        shortcut="F5",
        tooltip="Scan library directories for changes"
    )
    
    registry.register_action(
        "tools.reprocess",
        "&Reprocess Selection",
        window.reprocess_selection,
        shortcut="F6",
        tooltip="Run Phase 2/3 processing on selected files (metadata, embeddings, detections)"
    )
    
    registry.register_action(
        "tools.reindex_all",
        "Reindex &All Files",
        window.reindex_all_files,
        shortcut="F7",
        tooltip="Reprocess all unprocessed files in database (background)"
    )
    
    registry.register_action(
        "tools.library",
        "&Library Settings...",
        window.show_library_dialog,
        tooltip="Configure library roots and scan settings"
    )
    
    registry.register_action(
        "tools.rules",
        "&Rules Manager...",
        window.show_rules_dialog,
        tooltip="Manage automation rules"
    )
    
    registry.register_action(
        "tools.command_palette",
        "&Command Palette...",
        window.show_command_palette,
        shortcut="Ctrl+Shift+P",
        tooltip="Search and execute commands"
    )
    
    # ============================================================
    # MAINTENANCE MENU
    # ============================================================
    
    registry.register_action(
        "maintenance.rebuild_counts",
        "🔄 &Rebuild All Counts...",
        window.rebuild_all_counts,
        tooltip="Recalculate file counts for tags, albums, and directories"
    )
    
    registry.register_action(
        "maintenance.verify_references",
        "🔍 &Verify References...",
        window.verify_references,
        tooltip="Check for broken ObjectId references in database"
    )
    
    registry.register_action(
        "maintenance.cleanup_orphaned",
        "🧹 &Cleanup Orphaned Records...",
        window.cleanup_orphaned_records,
        tooltip="Remove references to deleted records"
    )
    
    # ============================================================
    # HELP MENU
    # ============================================================
    
    registry.register_action(
        "help.shortcuts",
        "&Keyboard Shortcuts...",
        window.show_shortcuts_dialog,
        shortcut="Ctrl+?",
        tooltip="Show keyboard shortcuts reference"
    )
    
    registry.register_action(
        "help.about",
        "&About UExplorer",
        window.show_about_dialog,
        tooltip="About UExplorer"
    )
    
    # ============================================================
    # SYNC PANEL CHECKABLE STATES
    # ============================================================
    
    # Set initial checked states for panel toggles
    if hasattr(window, 'docking_service') and window.docking_service:
        _sync_panel_states(registry, window)


def _sync_panel_states(registry: 'ActionRegistry', window: 'MainWindow'):
    """Sync panel toggle action states with actual panel visibility."""
    panel_names = ["tags", "albums", "directories", "relations", "properties", "filters", "search", "background", "similar", "annotation", "maintenance"]
    
    for name in panel_names:
        action_name = f"view.panel.{name}"
        # Check actual visibility from docking_service
        is_visible = window.docking_service.is_panel_visible(name)
        registry.set_checked(action_name, is_visible)

