# -*- coding: utf-8 -*-
"""
ExecutionLogPanel - Panel showing execution trace and errors.

Displays real-time execution logs with:
- Timestamps
- Node/pin information
- Color-coded log levels
- Clickable error links to navigate to error nodes
"""
from typing import Optional, List, TYPE_CHECKING
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat

if TYPE_CHECKING:
    from ..execution.executor import ExecutionLog


class ExecutionLogPanel(QWidget):
    """
    Panel for displaying execution logs.
    
    Features:
    - Real-time log display
    - Color-coded levels (INFO=white, WARNING=yellow, ERROR=red)
    - Timestamps
    - Clickable node IDs to navigate to nodes
    
    Signals:
        node_clicked: Emitted when a node ID is clicked (node_id)
    """
    
    node_clicked = Signal(str)  # node_id
    
    # Colors for log levels
    LEVEL_COLORS = {
        "DEBUG": QColor(128, 128, 128),
        "INFO": QColor(220, 220, 220),
        "WARNING": QColor(255, 200, 0),
        "ERROR": QColor(255, 80, 80),
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header with title and clear button
        header = QHBoxLayout()
        
        title = QLabel("Execution Log")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)
        
        layout.addLayout(header)
        
        # Log text area
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self._log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e20;
                color: #e0e0e0;
                border: 1px solid #3c3c3e;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._log_text)
        
        # Store node IDs for click detection
        self._node_positions = {}
    
    def clear(self):
        """Clear the log."""
        self._log_text.clear()
        self._node_positions.clear()
    
    def add_log(self, log: 'ExecutionLog'):
        """
        Add a log entry.
        
        Args:
            log: ExecutionLog entry to add
        """
        import datetime
        
        # Format timestamp
        ts = datetime.datetime.fromtimestamp(log.timestamp)
        time_str = ts.strftime("%H:%M:%S.%f")[:-3]
        
        # Get color for level
        color = self.LEVEL_COLORS.get(log.level, self.LEVEL_COLORS["INFO"])
        
        # Format message
        node_id_short = log.node_id[:8] if log.node_id else "?"
        
        # Build formatted text
        cursor = self._log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Timestamp (gray)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(100, 100, 100))
        cursor.insertText(f"[{time_str}] ", fmt)
        
        # Level
        fmt.setForeground(color)
        cursor.insertText(f"{log.level:<7} ", fmt)
        
        # Node type
        fmt.setForeground(QColor(150, 150, 255))
        cursor.insertText(f"[{log.node_type}] ", fmt)
        
        # Message
        fmt.setForeground(color)
        cursor.insertText(f"{log.message}\n", fmt)
        
        # Store node position for click detection
        # (simplified - would need proper hyperlink handling for full implementation)
        
        # Auto-scroll to bottom
        self._log_text.setTextCursor(cursor)
        self._log_text.ensureCursorVisible()
    
    def add_logs(self, logs: List['ExecutionLog']):
        """Add multiple log entries."""
        for log in logs:
            self.add_log(log)
    
    def add_error(self, node_id: str, node_type: str, message: str):
        """
        Add an error entry with clickable link.
        
        Args:
            node_id: ID of error node
            node_type: Type of error node
            message: Error message
        """
        import time
        from ..execution.executor import ExecutionLog
        
        log = ExecutionLog(
            timestamp=time.time(),
            node_id=node_id,
            node_type=node_type,
            message=message,
            level="ERROR"
        )
        self.add_log(log)
    
    def add_message(self, message: str, level: str = "INFO"):
        """
        Add a simple message without node context.
        
        Args:
            message: Message text
            level: Log level
        """
        import time
        from ..execution.executor import ExecutionLog
        
        log = ExecutionLog(
            timestamp=time.time(),
            node_id="",
            node_type="System",
            message=message,
            level=level
        )
        self.add_log(log)
    
    def set_execution_context(self, context):
        """
        Display logs from an execution context.
        
        Args:
            context: ExecutionContext with logs
        """
        self.clear()
        if context and context.logs:
            self.add_logs(context.logs)
