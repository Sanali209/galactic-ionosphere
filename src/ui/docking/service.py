"""
Docking Service - MVVM-compatible wrapper for PySide6-QtAds.

Maintains MVVM separation by providing a service layer between
ViewModels and the docking system.

Features:
- Document management (center area, tabs)
- Panel management (side areas, auto-hide)
- Perspectives (named layout configurations)
- Full state persistence support
- Configuration via flags

Usage:
    # Configure before creating (optional)
    DockingService.set_config_flag("FocusHighlighting", True)
    
    # Create service
    docking_service = DockingService(main_window)
    
    # Add documents (center content area)
    docking_service.add_document("doc1", my_widget, "Document 1")
    
    # Add panels (side areas)
    docking_service.add_panel("properties", props_widget, "Properties", area="right")
    
    # Perspectives
    docking_service.save_perspective("default")
    docking_service.load_perspective("coding")
"""
from typing import Optional, Dict, List, Any, Literal
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
import PySide6QtAds as QtAds
from loguru import logger


DockType = Literal["document", "panel"]
DockArea = Literal["center", "left", "right", "top", "bottom"]
SideBar = Literal["left", "right", "top", "bottom"]


class DockingService(QObject):
    """
    MVVM-compatible docking service for documents AND panels.
    
    ViewModels interact with this service, not directly with QtAds.
    Separates business logic from UI presentation.
    
    Documents are main content (editors, viewers) shown in the center with tabs.
    Panels are tool windows (properties, explorer) shown on edges.
    
    Attributes:
        dock_manager: The underlying QtAds CDockManager instance
        
    Signals:
        document_opened(str): Emitted when a document is added
        document_closed(str): Emitted when a document is closed
        document_activated(str): Emitted when a document tab is activated
        panel_opened(str): Emitted when a panel is added
        panel_closed(str): Emitted when a panel is closed
        panel_visibility_changed(str, bool): Emitted when panel visibility changes
        layout_changed(): Emitted when layout is restored
        perspective_changed(str): Emitted when perspective changes
        focus_changed(str): Emitted when focused dock changes
    """
    
    # Signals for MVVM pattern
    document_opened = Signal(str)  # doc_id
    document_closed = Signal(str)
    document_activated = Signal(str)
    panel_opened = Signal(str)  # panel_id
    panel_closed = Signal(str)
    panel_visibility_changed = Signal(str, bool)  # panel_id, visible
    layout_changed = Signal()
    perspective_changed = Signal(str)  # perspective_name
    focus_changed = Signal(str)  # dock_id
    
    def _on_focused_dock_widget_changed(self, old: QtAds.CDockWidget, new: QtAds.CDockWidget):
        """
        Handle unified focus change (documents OR panels).
        
        This is CRITICAL for keeping track of the active context.
        """
        if not new:
             return
             
        # Find ID for this dock widget
        dock_id = self._dock_to_id.get(new)
        
        if dock_id:
            logger.debug(f"Focus changed to dock: {dock_id}")
            self.focus_changed.emit(dock_id)
            
            # If it's a document, also emit document_activated for compatibility
            if dock_id in self._documents:
                self.document_activated.emit(dock_id)
        else:
            # Might be a dock widget we don't track (e.g. internal wrapper)
            pass
    
    # === Class-level Configuration ===
    
    @classmethod
    def set_config_flag(cls, flag: str, enabled: bool = True) -> None:
        """
        Set a global configuration flag BEFORE creating instance.
        
        Args:
            flag: Flag name (e.g., "FocusHighlighting", "MiddleMouseButtonClosesTab")
            enabled: Enable or disable the flag
        
        Available flags:
            - OpaqueSplitterResize: Dynamic splitter resizing
            - DragPreviewIsDynamic: Dynamic drag preview
            - DragPreviewShowsContentPixmap: Show content in drag preview
            - DragPreviewHasWindowFrame: Drag preview has window frame
            - FocusHighlighting: Highlight focused dock
            - MiddleMouseButtonClosesTab: Close tab with middle click
            - DockAreaHasCloseButton: Show close button on dock areas
            - DockAreaHasUndockButton: Show undock button on dock areas
            - DockAreaHasTabsMenuButton: Show tabs menu button
            - AlwaysShowTabs: Always show tabs even for single widget
        """
        flag_enum = getattr(QtAds.CDockManager, flag, None)
        if flag_enum is not None:
            QtAds.CDockManager.setConfigFlag(flag_enum, enabled)
            logger.debug(f"DockingService config flag set: {flag} = {enabled}")
        else:
            logger.warning(f"Unknown DockingService config flag: {flag}")
    
    @classmethod
    def set_config_preset(cls, preset: str) -> None:
        """
        Apply a preset configuration.
        
        Args:
            preset: "default_opaque" or "default_non_opaque"
        """
        if preset == "default_opaque":
            QtAds.CDockManager.setConfigFlags(QtAds.CDockManager.DefaultOpaqueConfig)
        elif preset == "default_non_opaque":
            QtAds.CDockManager.setConfigFlags(QtAds.CDockManager.DefaultNonOpaqueConfig)
        else:
            logger.warning(f"Unknown config preset: {preset}")
    
    def __init__(self, parent_widget: QWidget):
        """
        Initialize the docking service.
        
        Args:
            parent_widget: The parent QMainWindow or QWidget for the dock manager
        """
        super().__init__()
        
        # Configure QtAds BEFORE creating dock manager (critical!)
        self._configure_manager()
        
        # Now create the dock manager
        self.dock_manager = QtAds.CDockManager(parent_widget)
        
        # Separate tracking for documents vs panels
        self._documents: Dict[str, QtAds.CDockWidget] = {}
        self._panels: Dict[str, QtAds.CDockWidget] = {}
        
        # Track widget to ID mapping for signals
        self._dock_to_id: Dict[QtAds.CDockWidget, str] = {}
        
        # Parallel Python-side tracking to avoid C++ object deletion issues
        # Stores {id: {'widget': QWidget, 'title': str}}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Connect focus tracking
        self.dock_manager.focusedDockWidgetChanged.connect(self._on_focused_dock_widget_changed)
        
        # Connect robust cleanup handler
        # Use available signal (differs by QtAds version)
        if hasattr(self.dock_manager, 'dockWidgetAboutToBeRemoved'):
            self.dock_manager.dockWidgetAboutToBeRemoved.connect(self._on_dock_removed)
        elif hasattr(self.dock_manager, 'dockWidgetAboutToBeClosed'):
            self.dock_manager.dockWidgetAboutToBeClosed.connect(self._on_dock_removed)
            
        logger.info("DockingService initialized (documents + panels)")
    
    @property
    def documents(self) -> Dict[str, QtAds.CDockWidget]:
        """Get active documents."""
        return self._documents
    
    @property
    def panels(self) -> Dict[str, QtAds.CDockWidget]:
        """Get all panels."""
        return self._panels
    
    def _configure_manager(self):
        """
        Configure QtAds features - MUST be called BEFORE creating CDockManager.
        
        See QtAds docs: Config flags are global and must be set before
        instantiating the first CDockManager.
        """
        # Helper to safely set config flags
        def safe_set_flag(flag_name: str, value: bool = True):
            if hasattr(QtAds.CDockManager, flag_name):
                QtAds.CDockManager.setConfigFlag(
                    getattr(QtAds.CDockManager, flag_name), value
                )
                logger.debug(f"Config flag set: {flag_name} = {value}")
        
        # Enable dock area buttons
        safe_set_flag('DockAreaHasCloseButton', True)
        safe_set_flag('DockAreaHasUndockButton', True)
        safe_set_flag('DockAreaHasTabsMenuButton', True)
        
        # Optional visual features
        safe_set_flag('MiddleMouseButtonClosesTab', True)
        safe_set_flag('FocusHighlighting', True)
    
    # === Document Management ===
    
    def add_document(self, doc_id: str, widget: QWidget, title: str,
                    area: DockArea = "center", closable: bool = True,
                    delete_on_close: bool = False) -> None:
        """
        Add a DOCUMENT (main content).
        
        Documents go in center area, show in tabs.
        
        Args:
            doc_id: Unique document identifier
            widget: The widget to display
            title: Tab title
            area: Dock area ("center", "left", "right", "top", "bottom")
            closable: Whether user can close this document
            delete_on_close: If True, widget is deleted when closed (dynamic documents)
        """
        if doc_id in self._documents:
            # Check if it's a zombie
            existing = self._documents[doc_id]
            is_valid = False
            try:
                if existing and isinstance(existing, QtAds.CDockWidget) and existing.widget():
                    is_valid = True
            except RuntimeError:
                pass
                
            if is_valid:
                logger.warning(f"Document {doc_id} already exists, activating instead")
                self.activate_document(doc_id)
                return
            else:
                logger.warning(f"Document {doc_id} exists but is zombie - OVERWRITING")
                del self._documents[doc_id]

        
        dock = QtAds.CDockWidget(title)
        dock.setObjectName(doc_id)  # CRITICAL: Required for restoreState matching
        dock.setWidget(widget)
        
        # Document-specific features
        dock.setFeature(QtAds.CDockWidget.DockWidgetClosable, closable)
        dock.setFeature(QtAds.CDockWidget.DockWidgetMovable, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetFloatable, True)
        
        # Delete on close for dynamic documents (if available in this QtAds version)
        if delete_on_close:
            if hasattr(QtAds.CDockWidget, 'DockWidgetDeleteOnClose'):
                dock.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
            else:
                logger.debug("DockWidgetDeleteOnClose not available in this QtAds version")
        
        # Connect view toggle (when tab clicked) to document_activated
        dock.viewToggled.connect(
            lambda visible, did=doc_id: self._on_doc_view_toggled(did, visible)
        )
        
        # Connect close event
        dock.closed.connect(
            lambda did=doc_id: self._on_document_closed(did)
        )
        
        # Add to docking system - ensure proper center placement
        area_enum = self._get_area_enum(area)
        
        # For center documents, try to add as tab to existing document's area
        if area == "center" and self._documents:
            # Find an existing visible document to tab with
            for existing_dock in self._documents.values():
                try:
                    existing_area = existing_dock.dockAreaWidget()
                    if existing_area:
                        self.dock_manager.addDockWidget(area_enum, dock, existing_area)
                        break
                except RuntimeError:
                    # C++ object deleted, try next document
                    continue
            else:
                # No valid existing area found, add normally
                self.dock_manager.addDockWidget(area_enum, dock)
        else:
            # First document or non-center - add normally
            self.dock_manager.addDockWidget(area_enum, dock)
        
        self._documents[doc_id] = dock
        self._dock_to_id[dock] = doc_id
        
        # Store metadata for persistence safety (survives C++ deletion)
        self._metadata[doc_id] = {'widget': widget, 'title': title, 'is_document': True}
        
        self.document_opened.emit(doc_id)
        logger.info(f"Document added: {doc_id} - {title}")
    
    def _on_document_closed(self, doc_id: str) -> None:
        """Handle document close event."""
        if doc_id in self._documents:
            dock = self._documents.pop(doc_id, None)
            if dock:
                self._dock_to_id.pop(dock, None)
            self.document_closed.emit(doc_id)
            logger.debug(f"Document closed: {doc_id}")
    
    def get_document_widget(self, doc_id: str) -> Optional[QWidget]:
        """Get the widget for a document."""
        if doc_id in self._documents:
            return self._documents[doc_id].widget()
        return None
    
    def get_all_document_ids(self) -> List[str]:
        """Get list of all open document IDs."""
        return list(self._documents.keys())
    
    def get_active_document_id(self) -> Optional[str]:
        """Get the currently active/focused document ID."""
        for doc_id, dock in self._documents.items():
            try:
                # Safeguard against deleted C++ objects
                if not dock or not isinstance(dock, QtAds.CDockWidget):
                    continue
                    
                if dock.isCurrentTab():
                    return doc_id
            except RuntimeError:
                continue
        return None
    
    def set_document_title(self, doc_id: str, title: str) -> None:
        """Update document tab title."""
        if doc_id in self._metadata:
             self._metadata[doc_id]['title'] = title
             
        if doc_id in self._documents:
            try:
                self._documents[doc_id].setWindowTitle(title)
            except RuntimeError:
                pass
    
    # === Cleanup Logic ===
    
    def _on_dock_removed(self, dock: QtAds.CDockWidget):
        """Handle dock closing/removal at the Manager level (most robust)."""
        if not dock:
            return
            
        # Check documents
        if dock in self._dock_to_id:
            uid = self._dock_to_id[dock]
            logger.debug(f"Dock removed: {uid}")
            
            # Remove from dock tracking, but KEEP metadata for session persistence
            # Metadata is needed until session.save() completes after closeEvent
            if uid in self._documents and self._documents[uid] == dock:
                self._documents.pop(uid)
                self.document_closed.emit(uid)
                # DO NOT remove metadata here - needed for get_document_states
            elif uid in self._panels and self._panels[uid] == dock:
                self._panels.pop(uid)
                self.panel_closed.emit(uid)
                # DO NOT remove panel metadata either
                
            del self._dock_to_id[dock]

    # === Panel Management ===
    
    def add_panel(self, panel_id: str, widget: QWidget, title: str,
                 area: DockArea = "right", 
                 auto_hide: bool = False,
                 closable: bool = False) -> None:
        """
        Add a PANEL (tool/supporting view).
        
        Panels go to sides, can auto-hide, usually not closable.
        
        Args:
            panel_id: Unique panel identifier
            widget: The widget to display
            title: Panel title
            area: Dock area (default: "right")
            auto_hide: Whether panel starts auto-hidden
            closable: Whether user can close this panel
        """
        if panel_id in self._panels:
            logger.warning(f"Panel {panel_id} already exists")
            return
        
        dock = QtAds.CDockWidget(title)
        dock.setObjectName(panel_id)  # CRITICAL: Required for restoreState matching
        dock.setWidget(widget)
        
        # Panel-specific features
        dock.setFeature(QtAds.CDockWidget.DockWidgetClosable, closable)
        dock.setFeature(QtAds.CDockWidget.DockWidgetMovable, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetFloatable, True)
        
        # Pinnable feature (if available in this QtAds version)
        if hasattr(QtAds.CDockWidget, 'DockWidgetPinnable'):
            dock.setFeature(QtAds.CDockWidget.DockWidgetPinnable, True)
        
        # Add to docking system
        area_enum = self._get_area_enum(area)
        
        if auto_hide:
            # Add as auto-hide widget
            sidebar = self._get_sidebar_enum(area)
            self.dock_manager.addAutoHideDockWidget(sidebar, dock)
            logger.debug(f"Panel {panel_id} added as auto-hide to {area}")
        else:
            self.dock_manager.addDockWidget(area_enum, dock)
        
        # Connect visibility changes
        dock.viewToggled.connect(
            lambda visible, pid=panel_id: self.panel_visibility_changed.emit(pid, visible)
        )
        
        self._panels[panel_id] = dock
        self._dock_to_id[dock] = panel_id
        
        # Store metadata (is_document=False to distinguish from documents)
        self._metadata[panel_id] = {'widget': widget, 'title': title, 'is_document': False}
        
        self.panel_opened.emit(panel_id)
        logger.info(f"Panel added: {panel_id} - {title} (area={area}, auto_hide={auto_hide})")
    
    def set_panel_auto_hide(self, panel_id: str, enabled: bool,
                           sidebar: SideBar = "right") -> None:
        """
        Enable/disable auto-hide for a panel.
        
        Args:
            panel_id: Panel to modify
            enabled: Enable or disable auto-hide
            sidebar: Which sidebar (left, right, top, bottom)
        """
        if panel_id not in self._panels:
            logger.warning(f"Panel not found: {panel_id}")
            return
        
        dock = self._panels[panel_id]
        
        if enabled:
            sidebar_enum = self._get_sidebar_enum(sidebar)
            self.dock_manager.addAutoHideDockWidget(sidebar_enum, dock)
            logger.info(f"Panel {panel_id} set to auto-hide ({sidebar})")
        else:
            dock.setAutoHide(False)
            logger.info(f"Panel {panel_id} auto-hide disabled")
    
    def toggle_panel(self, panel_id: str) -> None:
        """Toggle panel visibility."""
        if panel_id in self._panels:
            self._panels[panel_id].toggleView()
    
    def show_panel(self, panel_id: str) -> None:
        """Show a panel."""
        if panel_id in self._panels:
            self._panels[panel_id].toggleView(True)
    
    def hide_panel(self, panel_id: str) -> None:
        """Hide a panel."""
        if panel_id in self._panels:
            self._panels[panel_id].toggleView(False)
    
    def is_panel_visible(self, panel_id: str) -> bool:
        """Check if panel is visible."""
        if panel_id in self._panels:
            return not self._panels[panel_id].isClosed()
        return False
    
    def get_panel_widget(self, panel_id: str) -> Optional[QWidget]:
        """Get the widget for a panel."""
        if panel_id in self._panels:
            return self._panels[panel_id].widget()
        return None
    
    # === Document Operations ===
    
    def close_document(self, doc_id: str) -> None:
        """Close a document."""
        if doc_id in self._documents:
            self._documents[doc_id].closeDockWidget()
            # Note: _on_document_closed will handle cleanup
    
    def activate_document(self, doc_id: str) -> None:
        """Bring document to front."""
        if doc_id in self._documents:
            self._documents[doc_id].toggleView(True)
            self._documents[doc_id].raise_()
            self.document_activated.emit(doc_id)
    
    def _on_doc_view_toggled(self, doc_id: str, visible: bool):
        """Handle document view toggle - emit activation when visible."""
        if visible:
            self.document_activated.emit(doc_id)
            logger.debug(f"Document activated via tab: {doc_id}")
    
    # === Perspectives ===
    
    def save_perspective(self, name: str) -> None:
        """
        Save current layout as a named perspective.
        
        Args:
            name: Perspective name
        """
        self.dock_manager.addPerspective(name)
        self.perspective_changed.emit(name)
        logger.info(f"Perspective saved: {name}")
    
    def load_perspective(self, name: str) -> bool:
        """
        Load a named perspective.
        
        Args:
            name: Perspective to load
            
        Returns:
            True if loaded successfully
        """
        if name in self.dock_manager.perspectiveNames():
            self.dock_manager.openPerspective(name)
            self.perspective_changed.emit(name)
            self.layout_changed.emit()
            logger.info(f"Perspective loaded: {name}")
            return True
        else:
            logger.warning(f"Perspective not found: {name}")
            return False
    
    def get_perspectives(self) -> List[str]:
        """Get list of available perspective names."""
        try:
            return list(self.dock_manager.perspectiveNames())
        except RuntimeError:
            logger.debug("Dock manager deleted, cannot get perspectives")
            return []
    
    def remove_perspective(self, name: str) -> None:
        """Remove a saved perspective."""
        self.dock_manager.removePerspective(name)
        logger.info(f"Perspective removed: {name}")
    
    # === Layout Persistence ===
    
    def save_layout(self) -> bytes:
        """Save layout including panel states."""
        try:
            return self.dock_manager.saveState()
        except RuntimeError:
            logger.warning("Dock manager deleted, cannot save layout state")
            return b""
    
    def restore_layout(self, state: bytes) -> None:
        """Restore layout."""
        self.dock_manager.restoreState(state)
        self.layout_changed.emit()
    
    # === State Collection for Session Persistence ===
    
    def get_document_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state of all documents for persistence.
        
        Uses the Python-side _metadata dict which survives C++ object deletion.
        
        Returns:
            Dict mapping doc_id to state dict
        """
        states = {}
        
        # Use metadata which contains Python widget references that survive C++ deletion
        for doc_id, meta in self._metadata.items():
            # Skip internal placeholder document
            if doc_id == "__placeholder__":
                continue
                
            # Check if this was a document (not a panel) using stored flag
            if not meta.get('is_document', False):
                continue
                
            widget = meta.get('widget')
            title = meta.get('title', 'Unknown')
            
            # Try to get dock properties, but have fallbacks
            is_active = False
            is_floating = False
            is_closed = True  # Default to closed since dock may be gone
            
            dock = self._documents.get(doc_id)
            if dock:
                try:
                    is_active = dock.isCurrentTab()
                    is_floating = dock.isFloating()
                    is_closed = dock.isClosed()
                except RuntimeError:
                    # C++ object deleted, use defaults
                    pass
            
            try:
                states[doc_id] = {
                    "title": title,
                    "is_active": is_active,
                    "is_floating": is_floating,
                    "is_closed": is_closed,
                    "custom_state": widget.get_state() if widget and hasattr(widget, 'get_state') else {}
                }
                logger.debug(f"Collected state for document {doc_id}")
            except Exception as e:
                logger.debug(f"Error collecting state for {doc_id}: {e}")
                
        return states
    
    def get_panel_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state of all panels for persistence.
        
        Returns:
            Dict mapping panel_id to state dict
        """
        states = {}
        for panel_id, dock in self._panels.items():
            try:
                # Check safeguards for deleted C++ object
                if not dock or not isinstance(dock, QtAds.CDockWidget):
                   continue

                try:
                    widget = dock.widget()
                    is_auto_hide = dock.isAutoHide() if hasattr(dock, 'isAutoHide') else False
                    
                    states[panel_id] = {
                        "title": dock.windowTitle(),
                        "is_visible": not dock.isClosed(),
                        "is_floating": dock.isFloating(),
                        "is_auto_hide": is_auto_hide,
                        "custom_state": widget.get_state() if hasattr(widget, 'get_state') else {}
                    }
                except RuntimeError:
                     logger.debug(f"Panel {panel_id} widget already deleted - skipping state collection")
                     continue
                     
            except Exception as e:
                logger.debug(f"Error collecting state for {panel_id}: {e}")
                
        return states
    
    def get_complete_state(self) -> Dict[str, Any]:
        """
        Get complete docking state for session persistence.
        
        Returns:
            Dict with all docking state
        """
        try:
            layout_bytes = self.save_layout()
            # QByteArray doesn't have .hex() - convert to Python bytes first
            if layout_bytes:
                layout_hex = bytes(layout_bytes).hex()
            else:
                layout_hex = None
        except Exception as e:
            logger.warning(f"Failed to save layout bytes: {e}")
            layout_hex = None
        
        return {
            "layout_bytes": layout_hex,
            "documents": self.get_document_states(),
            "panels": self.get_panel_states(),
            "perspectives": self.get_perspectives(),
            "active_document": self.get_active_document_id(),
        }
    
    # === Helpers ===
    
    def _get_area_enum(self, area: DockArea):
        """Convert area string to QtAds enum."""
        area_map = {
            "center": QtAds.CenterDockWidgetArea,
            "left": QtAds.LeftDockWidgetArea,
            "right": QtAds.RightDockWidgetArea,
            "top": QtAds.TopDockWidgetArea,
            "bottom": QtAds.BottomDockWidgetArea,
        }
        return area_map.get(area, QtAds.CenterDockWidgetArea)
    
    def _get_sidebar_enum(self, sidebar: SideBar):
        """Convert sidebar string to QtAds SideBarLocation enum."""
        sidebar_map = {
            "left": QtAds.SideBarLeft,
            "right": QtAds.SideBarRight,
            "top": QtAds.SideBarTop,
            "bottom": QtAds.SideBarBottom,
        }
        return sidebar_map.get(sidebar, QtAds.SideBarRight)
