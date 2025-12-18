"""
Output panel for displaying application logs.
"""
from PySide6.QtWidgets import QVBoxLayout, QTextEdit
from PySide6.QtCore import Slot
from .panel_base import BasePanelWidget

class OutputPanel(BasePanelWidget):
    """
    Displays application logs and messages.
    """
    def __init__(self, title: str, locator, parent=None):
        super().__init__(title, locator, parent)
        self.max_lines = 1000
    
    def initialize_ui(self):
        """Build the output panel UI."""
        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        
        layout.addWidget(self.text_edit)
    
    @Slot(str)
    def append_message(self, message: str):
        """Add a message to the output."""
        self.text_edit.append(message)
        
        # Limit number of lines
        if self.text_edit.document().lineCount() > self.max_lines:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # Delete newline
    
    def clear(self):
        """Clear all output."""
        self.text_edit.clear()
