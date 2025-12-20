"""
Dock manager for panel lifecycle and state persistence.
"""
from typing import Dict, Type, Optional
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt
from loguru import logger
from uexplorer_src.ui.docking.panel_base import BasePanelWidget

class PanelState:
    """Represents saved state of a panel."""
    def __init__(self, visible: bool = True, area: str = "right", 
                 width: int = 300, height: int = 200):
        self.visible = visible
        self.area = area  # "left", "right", "top", "bottom", "floating"
        self.width = width
        self.height = height
        self.custom_state = {}
    
    def to_dict(self) -> dict:
        return {
            "visible": self.visible,
            "area": self.area,
            "width": self.width,
            "height": self.height,
            "custom_state": self.custom_state
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PanelState':
        state = cls(
            visible=data.get("visible", True),
            area=data.get("area", "right"),
            width=data.get("width", 300),
            height=data.get("height", 200)
        )
        state.custom_state = data.get("custom_state", {})
        return state

class DockManager:
    """
    Manages all dockable panels.
    Handles registration, creation, and state persistence.
    """
    def __init__(self, main_window: QMainWindow, locator):
        self.main_window = main_window
        self.locator = locator  # ServiceLocator for service access
        self._panel_registry: Dict[str, Type[BasePanelWidget]] = {}
        self._active_panels: Dict[str, BasePanelWidget] = {}
        logger.info("DockManager initialized")
    
    def register_panel(self, name: str, panel_class: Type[BasePanelWidget]):
        """
        Register a panel type.
        name: unique identifier (e.g., "properties", "output")
        panel_class: class that inherits from BasePanelWidget
        """
        self._panel_registry[name] = panel_class
        logger.debug(f"Panel registered: {name}")
    
    def create_panel(self, name: str) -> Optional[BasePanelWidget]:
        """
        Create and show a panel by name.
        Returns the panel instance.
        """
        if name in self._active_panels:
            # Panel already exists, just show it
            panel = self._active_panels[name]
            panel.show()
            return panel
        
        if name not in self._panel_registry:
            logger.error(f"Panel not registered: {name}")
            return None
        
        # Create new panel instance
        panel_class = self._panel_registry[name]
        
        # Pass locator to panel for service access
        panel = panel_class(name.replace('_', ' ').title(), 
                           self.locator, 
                           self.main_window)
        
        panel.initialize_ui()
        
        # Add to main window (default right side)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, panel)
        
        self._active_panels[name] = panel
        logger.info(f"Panel created: {name}")
        
        return panel
    
    def get_panel(self, name: str) -> Optional[BasePanelWidget]:
        """Get active panel by name."""
        return self._active_panels.get(name)
    
    def show_panel(self, name: str):
        """Show panel (create if doesn't exist)."""
        panel = self.get_panel(name) or self.create_panel(name)
        if panel:
            panel.show()
    
    def hide_panel(self, name: str):
        """Hide panel."""
        panel = self.get_panel(name)
        if panel:
            panel.hide()
    
    def toggle_panel(self, name: str):
        """Toggle panel visibility."""
        panel = self.get_panel(name)
        if panel:
            if panel.isVisible():
                panel.hide()
            else:
                panel.show()
        else:
            self.create_panel(name)
    
    def save_state(self) -> dict:
        """
        Save state of all panels.
        Returns dict suitable for saving to config.
        """
        state = {}
        
        for name, panel in self._active_panels.items():
            panel_state = PanelState()
            panel_state.visible = panel.isVisible()
            
            # Determine dock area
            area = self.main_window.dockWidgetArea(panel)
            area_map = {
                Qt.LeftDockWidgetArea: "left",
                Qt.RightDockWidgetArea: "right",
                Qt.TopDockWidgetArea: "top",
                Qt.BottomDockWidgetArea: "bottom"
            }
            panel_state.area = area_map.get(area, "right")
            
            # Get size
            panel_state.width = panel.width()
            panel_state.height = panel.height()
            
            # Custom panel state
            panel_state.custom_state = panel.get_state()
            
            state[name] = panel_state.to_dict()
        
        logger.info(f"Saved state for {len(state)} panels")
        return state
    
    def restore_state(self, state: dict):
        """
        Restore panel states from config.
        """
        for name, panel_data in state.items():
            panel_state = PanelState.from_dict(panel_data)
            
            # Create panel
            panel = self.create_panel(name)
            if not panel:
                continue
            
            # Restore position
            area_map = {
                "left": Qt.LeftDockWidgetArea,
                "right": Qt.RightDockWidgetArea,
                "top": Qt.TopDockWidgetArea,
                "bottom": Qt.BottomDockWidgetArea
            }
            if panel_state.area in area_map:
                self.main_window.addDockWidget(area_map[panel_state.area], panel)
            
            # Restore size
            panel.resize(panel_state.width, panel_state.height)
            
            # Restore visibility
            if panel_state.visible:
                panel.show()
            else:
                panel.hide()
            
            # Restore custom state
            panel.set_state(panel_state.custom_state)
        
        logger.info(f"Restored state for {len(state)} panels")
