"""
Command Palette - fuzzy search for all commands.
"""
from typing import List, Tuple
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QListWidget, 
                                QListWidgetItem, QLabel)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from loguru import logger

class CommandPalette(QDialog):
    """
    VS Code-style command palette.
    Fuzzy search through all registered commands.
    """
    command_selected = Signal(str)  # command name
    
    def __init__(self, action_registry, parent=None):
        super().__init__(parent)
        self.actions = action_registry
        self.setWindowTitle("Command Palette")
        self.setModal(True)
        self.resize(600, 400)
        
        self._init_ui()
        self._populate_commands()
        
        logger.debug("CommandPalette initialized")
    
    def _init_ui(self):
        """Build command palette UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type a command...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)
        
        # Command list
        self.command_list = QListWidget()
        self.command_list.itemActivated.connect(self._on_command_activated)
        layout.addWidget(self.command_list)
        
        # Info label
        self.info_label = QLabel("Press Enter to execute, Esc to cancel")
        self.info_label.setStyleSheet("padding: 5px; background: #f0f0f0;")
        layout.addWidget(self.info_label)
    
    def _populate_commands(self):
        """Populate command list from action registry."""
        self.all_commands: List[Tuple[str, str]] = []
        
        for name, action in self.actions._actions.items():
            # Get display text and shortcut
            text = action.text().replace("&", "")  # Remove mnemonics
            shortcut = action.shortcut().toString()
            
            display = f"{text}"
            if shortcut:
                display += f" ({shortcut})"
            
            self.all_commands.append((name, display))
        
        self._update_list(self.all_commands)
    
    def _update_list(self, commands: List[Tuple[str, str]]):
        """Update the command list."""
        self.command_list.clear()
        
        for name, display in commands:
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            self.command_list.addItem(item)
        
        # Select first item
        if self.command_list.count() > 0:
            self.command_list.setCurrentRow(0)
    
    def _on_search(self, text: str):
        """Filter commands by search text."""
        if not text:
            self._update_list(self.all_commands)
            return
        
        # Simple fuzzy search
        text_lower = text.lower()
        filtered = [
            (name, display) 
            for name, display in self.all_commands
            if text_lower in display.lower()
        ]
        
        self._update_list(filtered)
    
    def _on_command_activated(self, item: QListWidgetItem):
        """Execute selected command."""
        command_name = item.data(Qt.UserRole)
        logger.info(f"Command palette: executing {command_name}")
        
        # Get action and trigger it
        action = self.actions.get_action(command_name)
        if action:
            action.trigger()
        
        self.accept()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses."""
        if event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            current = self.command_list.currentItem()
            if current:
                self._on_command_activated(current)
        elif event.key() == Qt.Key_Down:
            current_row = self.command_list.currentRow()
            if current_row < self.command_list.count() - 1:
                self.command_list.setCurrentRow(current_row + 1)
        elif event.key() == Qt.Key_Up:
            current_row = self.command_list.currentRow()
            if current_row > 0:
                self.command_list.setCurrentRow(current_row - 1)
        else:
            super().keyPressEvent(event)
    
    def showEvent(self, event):
        """Focus search input when shown."""
        super().showEvent(event)
        self.search_input.setFocus()
        self.search_input.selectAll()
