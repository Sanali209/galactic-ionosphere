from PySide6.QtCore import Qt, QLoggingCategory, QRegularExpression, QMimeData, Signal # Import Signal
from PySide6.QtGui import QAction, QTextCharFormat, QColor, QFont, QSyntaxHighlighter, QTextCursor, QFontMetrics, QDrag, QKeyEvent # Import QKeyEvent
from PySide6.QtWidgets import QTextEdit, QCompleter

class MongoQueryHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("blue"))
        self.keyword_format.setFontWeight(QFont.Bold)

        self.operators_format = QTextCharFormat()
        self.operators_format.setForeground(QColor("red"))

        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("green"))

        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("darkMagenta"))

        self.rules = [
            (
                QRegularExpression(r'\b(AND|OR|NOT|IN|NOT IN|ALL|SIZE|REGEX|TYPE|EXISTS|MOD|TEXT)\b'),
                self.keyword_format),
            (QRegularExpression(r'==|!=|>=|<=|>|<'), self.operators_format),
            (QRegularExpression(r'".*?"|\'.*?\''), self.string_format),
            (QRegularExpression(r'\b\d+\b'), self.number_format),
        ]

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)

class MongoQueryEditor(QTextEdit):
    # Define the custom signal
    returnPressed = Signal()

    def __init__(self):
        super().__init__()
        self.highlighter = MongoQueryHighlighter(self.document())
        self.completer = QCompleter([
            "AND", "OR", "NOT", "IN", "NOT IN", "ALL", "SIZE", "REGEX", "TYPE", "EXISTS", "MOD", "TEXT",
            "==", "!=", ">=", "<=", ">", "<", "true", "false",
            "name", "size", "tags", "local_path" # Add common field names
        ])
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        # Connect activated signal to insert completion
        self.completer.activated.connect(self.insert_completion)

    def keyPressEvent(self, event: QKeyEvent): # Add type hint for event
        is_completer_visible = self.completer.popup().isVisible()
        key = event.key()
        modifiers = event.modifiers()

        # Handle completer visibility shortcuts
        if key == Qt.Key_Space and modifiers == Qt.NoModifier:
            self.completer.popup().hide()
        elif key == Qt.Key_Space and modifiers == Qt.ControlModifier:
            # Manually trigger completion logic
            self._trigger_completion()
            event.accept() # Accept Ctrl+Space, don't insert space
            return

        # Handle Enter/Return press
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            if is_completer_visible:
                # Let the completer handle Enter (via activated signal)
                # We still ignore the event to prevent newline insertion by QTextEdit
                event.ignore()
                return
            elif not (modifiers & Qt.KeyboardModifier.ShiftModifier):
                # If completer is NOT visible and Shift is NOT pressed, emit signal
                self.returnPressed.emit()
                event.accept() # Accept the event, preventing newline insertion
                return
            # else: Shift+Enter allows newline, fall through to super().keyPressEvent

        # Handle other keys when completer is visible (Escape, Tab)
        if is_completer_visible and key in {Qt.Key_Escape, Qt.Key_Tab}:
            # Let completer handle Escape/Tab (hide popup or navigate)
            event.ignore()
            return

        # Call base implementation for other keys or Shift+Enter
        super().keyPressEvent(event)

        # Update completer prefix after handling the key press (if not handled above)
        # Only trigger completion if the popup is already visible or if it's a character key
        # (Avoid triggering on arrow keys, etc.)
        if is_completer_visible or (event.text() and not event.text().isspace()):
             self._trigger_completion()
        elif key == Qt.Key_Backspace: # Hide on backspace if popup not visible
             if not is_completer_visible:
                 self.completer.popup().hide()
             else: # Update completion if visible
                 self._trigger_completion()


    def _trigger_completion(self):
        """Helper method to update and show the completer."""
        cursor = self.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        prefix = cursor.selectedText()

        if prefix: # Only complete if there's a word under cursor
            self.completer.setCompletionPrefix(prefix)
            # Check if there are completions available for the prefix
            if self.completer.completionCount() > 0:
                rect = self.cursorRect()
                rect.setWidth(self.completer.popup().sizeHintForColumn(0) +
                              self.completer.popup().verticalScrollBar().sizeHint().width())
                self.completer.complete(rect) # Show or update the popup
            else:
                self.completer.popup().hide() # Hide if no completions match
        else:
            self.completer.popup().hide() # Hide if no word under cursor

    def insert_completion(self, completion):
        """Inserts the selected completion, replacing the current prefix."""
        cursor = self.textCursor()
        prefix_len = len(self.completer.completionPrefix())
        # Ensure we don't go beyond the start of the document
        pos = cursor.position()
        anchor = max(0, pos - prefix_len)
        cursor.setPosition(anchor, QTextCursor.MoveMode.MoveAnchor)
        cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)

        # Insert the selected completion
        cursor.insertText(completion)
        self.setTextCursor(cursor)
        # Hide completer after insertion
        self.completer.popup().hide()

    def focusOutEvent(self, event):
        """Hide completer when the editor loses focus."""
        self.completer.popup().hide()
        super().focusOutEvent(event)
