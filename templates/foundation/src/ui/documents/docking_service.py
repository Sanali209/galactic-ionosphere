"""
Docking Service - MVVM-compatible wrapper for PySide6-QtAds

Maintains MVVM separation by providing a service layer between
ViewModels and the docking system.
"""
from typing import Optional, Dict, Literal
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
import PySide6QtAds as QtAds
from loguru import logger


DockType = Literal["document", "panel"]
DockArea = Literal["center", "left", "right", "top", "bottom"]


class DockingService(QObject):
    """
    MVVM-compatible docking service for documents AND panels.
    
    ViewModels interact with this service, not directly with QtAds.
    Separates business logic from UI presentation.
    """
    
    # Signals for MVVM pattern
    document_opened = Signal(str)  # doc_id
    document_closed = Signal(str)
    document_activated = Signal(str)
    panel_opened = Signal(str)  # panel_id
    panel_closed = Signal(str)
    panel_visibility_changed = Signal(str, bool)  # panel_id, visible
    layout_changed = Signal()
    
    def __init__(self, parent_widget: QWidget):
        super().__init__()
        self.dock_manager = QtAds.CDockManager(parent_widget)
        
        # Separate tracking for documents vs panels
        self._documents: Dict[str, QtAds.CDockWidget] = {}
        self._panels: Dict[str, QtAds.CDockWidget] = {}
        
        # Configure QtAds
        self._configure_manager()
        
        logger.info("DockingService initialized (documents + panels)")
    
    def _configure_manager(self):
        """Configure QtAds features."""
        # Enable floating windows and dock area controls
        QtAds.CDockManager.setConfigFlag(
            QtAds.CDockManager.DockAreaHasCloseButton, True
        )
        QtAds.CDockManager.setConfigFlag(
            QtAds.CDockManager.DockAreaHasUndockButton, True
        )
        QtAds.CDockManager.setConfigFlag(
            QtAds.CDockManager.DockAreaHasTabsMenuButton, True
        )
        # Enable auto-hide for panels
        QtAds.CDockManager.setConfigFlag(
            QtAds.CDockManager.AllTabsHaveCloseButton, False
        )
    
    def add_document(self, doc_id: str, widget: QWidget, title: str,
                    area: DockArea = "center", closable: bool = True) -> None:
        """
        Add a DOCUMENT (main content).
        
        Documents go in center area, show in tabs.
        
        Args:
            doc_id: Unique document identifier
            widget: The widget to display
            title: Tab title
            area: Dock area ("center", "left", "right", "top", "bottom")
            closable: Whether user can close this document
        """
        dock = QtAds.CDockWidget(title)
        dock.setWidget(widget)
        
        # Document-specific features
        dock.setFeature(QtAds.CDockWidget.DockWidgetClosable, closable)
        dock.setFeature(QtAds.CDockWidget.DockWidgetMovable, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetFloatable, True)
        
        # Add to docking system
        area_enum = self._get_area_enum(area)
        self.dock_manager.addDockWidget(area_enum, dock)
        
        self._documents[doc_id] = dock
        self.document_opened.emit(doc_id)
        logger.info(f"Document added: {doc_id} - {title}")
    
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
        dock = QtAds.CDockWidget(title)
        dock.setWidget(widget)
        
        # Panel-specific features
        dock.setFeature(QtAds.CDockWidget.DockWidgetClosable, closable)
        dock.setFeature(QtAds.CDockWidget.DockWidgetMovable, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetFloatable, True)
        dock.setFeature(QtAds.CDockWidget.DockWidgetPinnable, True)  # Can pin
        
        # Add to docking system
        area_enum = self._get_area_enum(area)
        self.dock_manager.addDockWidget(area_enum, dock)
        
        # Note: Auto-hide implementation will be added later if needed
        # QtAds auto-hide requires more complex setup
        if auto_hide:
            logger.debug(f"Auto-hide requested for {panel_id} (not yet implemented)")
        
        # Connect visibility changes
        dock.viewToggled.connect(
            lambda visible, pid=panel_id: self.panel_visibility_changed.emit(pid, visible)
        )
        
        self._panels[panel_id] = dock
        self.panel_opened.emit(panel_id)
        logger.info(f"Panel added: {panel_id} - {title} (area={area}, auto_hide={auto_hide})")
    
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
    
    def close_document(self, doc_id: str) -> None:
        """Close a document."""
        if doc_id in self._documents:
            self._documents[doc_id].closeDockWidget()
            del self._documents[doc_id]
            self.document_closed.emit(doc_id)
    
    def activate_document(self, doc_id: str) -> None:
        """Bring document to front."""
        if doc_id in self._documents:
            self._documents[doc_id].toggleView(True)
            self._documents[doc_id].raise_()
            self.document_activated.emit(doc_id)
    
    def save_layout(self) -> bytes:
        """Save layout including panel states."""
        return self.dock_manager.saveState()
    
    def restore_layout(self, state: bytes) -> None:
        """Restore layout."""
        self.dock_manager.restoreState(state)
        self.layout_changed.emit()
    
    def _get_area_enum(self, area: DockArea):
        """Convert area string to QtAds enum."""
        area_map = {
            "center": QtAds.CenterDockWidgetArea,
            "left": QtAds.LeftDockWidgetArea,
            "right": QtAds.RightDockWidgetArea,
            "top": QtAds.TopDockWidgetArea,
            "bottom": QtAds.BottomDockWidgetArea,
        }
        return area_map[area]
