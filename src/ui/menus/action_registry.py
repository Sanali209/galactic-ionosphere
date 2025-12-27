"""
Action registry for managing QActions across the application.
"""
from typing import Dict, Callable, Optional
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget
from loguru import logger

class ActionRegistry:
    """
    Central registry for all QActions.
    Maps action names to QAction instances.
    """
    def __init__(self, parent: QWidget):
        self.parent = parent
        self._actions: Dict[str, QAction] = {}
        logger.info("ActionRegistry initialized")
    
    def register_action(self, 
                       name: str, 
                       text: str,
                       callback: Callable,
                       shortcut: Optional[str] = None,
                       tooltip: Optional[str] = None,
                       checkable: bool = False) -> QAction:
        """
        Register a new action.
        
        Args:
            name: Unique identifier (e.g., "file_open")
            text: Display text (e.g., "Open...")
            callback: Function to call when triggered
            shortcut: Keyboard shortcut (e.g., "Ctrl+O")
            tooltip: Status bar tooltip
            checkable: Whether action can be checked/unchecked
        
        Returns:
            QAction instance
        """
        if name in self._actions:
            logger.warning(f"Action already registered: {name}")
            return self._actions[name]
        
        action = QAction(text, self.parent)
        action.triggered.connect(callback)
        
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        
        if tooltip:
            action.setStatusTip(tooltip)
        
        action.setCheckable(checkable)
        
        self._actions[name] = action
        logger.debug(f"Action registered: {name}")
        
        return action
    
    def get_action(self, name: str) -> Optional[QAction]:
        """Get action by name."""
        return self._actions.get(name)
    
    def set_enabled(self, name: str, enabled: bool):
        """Enable/disable action."""
        action = self.get_action(name)
        if action:
            action.setEnabled(enabled)
    
    def set_checked(self, name: str, checked: bool):
        """Set checked state of checkable action."""
        action = self.get_action(name)
        if action and action.isCheckable():
            action.setChecked(checked)
    
    def update_text(self, name: str, text: str):
        """Update action text."""
        action = self.get_action(name)
        if action:
            action.setText(text)
