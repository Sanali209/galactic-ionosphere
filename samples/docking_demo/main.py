"""
Docking Demo - Sample application demonstrating Foundation's docking system.

Features demonstrated:
- DockingService with documents and panels
- Session persistence (documents restored on restart)
- Panel toggle buttons

Usage:
    python main.py
"""
import sys
import json
from pathlib import Path

# Add foundation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTextEdit, QLabel, QPushButton, QListWidget
)
from loguru import logger

from src.ui.docking import DockingService
from src.ui.documents import BaseDocumentWidget

# Session file path
SESSION_FILE = Path(__file__).parent / "session.json"


class TextEditorDocument(BaseDocumentWidget):
    """Sample document: Text editor."""
    
    def __init__(self, doc_id: str, title: str = "Untitled", parent=None):
        super().__init__(parent=parent)
        self._doc_id = doc_id
        self._title = title
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self._editor = QTextEdit()
        self._editor.setPlaceholderText(f"Type something in {title}...")
        layout.addWidget(self._editor)
    
    @property
    def doc_id(self) -> str:
        return self._doc_id
    
    def get_state(self) -> dict:
        return {"title": self._title, "content": self._editor.toPlainText()}
    
    def set_state(self, state: dict) -> None:
        self._title = state.get("title", "Untitled")
        self._editor.setPlainText(state.get("content", ""))


class PropertiesPanel(QWidget):
    """Sample panel: Properties viewer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Properties Panel</b>"))
        self._info = QLabel("No document selected")
        layout.addWidget(self._info)
        layout.addStretch()
    
    def set_document_info(self, info: str):
        self._info.setText(info)


class ExplorerPanel(QWidget):
    """Sample panel: File explorer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Explorer Panel</b>"))
        self._list = QListWidget()
        self._list.addItems(["Document 1", "Document 2", "Document 3"])
        layout.addWidget(self._list)


class OutputPanel(QWidget):
    """Sample panel: Output log."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Output Panel</b>"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)
    
    def log(self, message: str):
        self._log.append(message)


class DemoMainWindow(QMainWindow):
    """Main window with session persistence."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foundation Docking Demo")
        self.resize(1200, 800)
        
        # Create DockingService
        self.docking = DockingService(self)
        self.setCentralWidget(self.docking.dock_manager)
        
        # Track documents
        self._doc_count = 0
        
        # Create UI
        self._create_toolbar()
        self._create_panels()
        
        # Connect signals
        self.docking.document_activated.connect(self._on_document_activated)
        
        # Restore session OR create initial document
        if not self._restore_session():
            self._new_document()
        
        logger.info("DemoMainWindow initialized")
    
    def _create_toolbar(self):
        """Create toolbar with demo actions."""
        toolbar = self.addToolBar("Main")
        
        # New document button
        new_btn = QPushButton("+ New Document")
        new_btn.clicked.connect(self._new_document)
        toolbar.addWidget(new_btn)
        
        toolbar.addSeparator()
        
        # Panel toggles
        toggle_explorer = QPushButton("Toggle Explorer")
        toggle_explorer.clicked.connect(lambda: self.docking.toggle_panel("explorer"))
        toolbar.addWidget(toggle_explorer)
        
        toggle_properties = QPushButton("Toggle Properties")
        toggle_properties.clicked.connect(lambda: self.docking.toggle_panel("properties"))
        toolbar.addWidget(toggle_properties)
        
        toggle_output = QPushButton("Toggle Output")
        toggle_output.clicked.connect(lambda: self.docking.toggle_panel("output"))
        toolbar.addWidget(toggle_output)
    
    def _create_panels(self):
        """Create tool panels using DockingService."""
        self._explorer = ExplorerPanel()
        self.docking.add_panel("explorer", self._explorer, "Explorer", area="left")
        
        self._properties = PropertiesPanel()
        self.docking.add_panel("properties", self._properties, "Properties", area="right")
        
        self._output = OutputPanel()
        self.docking.add_panel("output", self._output, "Output", area="bottom")
    
    def _new_document(self):
        """Create a new text editor document."""
        self._doc_count += 1
        doc_id = f"doc_{self._doc_count}"
        title = f"Document {self._doc_count}"
        
        doc = TextEditorDocument(doc_id, title)
        self.docking.add_document(doc_id, doc, title, closable=True, delete_on_close=True)
        
        self._output.log(f"Created: {title}")
        logger.info(f"New document: {doc_id}")
    
    def _on_document_activated(self, doc_id: str):
        """Handle document activation."""
        self._properties.set_document_info(f"Active: {doc_id}")
        self._output.log(f"Activated: {doc_id}")
    
    # === Session Persistence ===
    
    def _save_session(self):
        """Save session to file."""
        state = self.docking.get_complete_state()
        
        # Update doc_count to max doc number
        for doc_id in state.get("documents", {}):
            try:
                num = int(doc_id.split("_")[1])
                self._doc_count = max(self._doc_count, num)
            except (IndexError, ValueError):
                pass
        
        session = {
            "doc_count": self._doc_count,
            "documents": state.get("documents", {}),
            "layout": state.get("layout_bytes"),
        }
        
        try:
            with open(SESSION_FILE, "w") as f:
                json.dump(session, f, indent=2)
            logger.info(f"Session saved: {len(session['documents'])} documents")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _restore_session(self) -> bool:
        """Restore session from file. Returns True if any documents restored."""
        if not SESSION_FILE.exists():
            return False
        
        try:
            with open(SESSION_FILE, "r") as f:
                session = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            return False
        
        # Restore doc_count
        self._doc_count = session.get("doc_count", 0)
        
        # Restore documents
        documents = session.get("documents", {})
        restored = 0
        
        for doc_id, doc_state in documents.items():
            try:
                title = doc_state.get("title", "Untitled")
                custom = doc_state.get("custom_state", {})
                
                doc = TextEditorDocument(doc_id, title)
                doc.set_state(custom)
                
                self.docking.add_document(doc_id, doc, title, closable=True, delete_on_close=True)
                self._output.log(f"Restored: {title}")
                restored += 1
            except Exception as e:
                logger.error(f"Failed to restore {doc_id}: {e}")
        
        # Restore layout if available
        layout_hex = session.get("layout")
        if layout_hex:
            try:
                layout_bytes = bytes.fromhex(layout_hex)
                self.docking.restore_layout(layout_bytes)
            except Exception as e:
                logger.warning(f"Failed to restore layout: {e}")
        
        logger.info(f"Session restored: {restored} documents")
        return restored > 0
    
    def closeEvent(self, event):
        """Save session on close."""
        self._save_session()
        event.accept()


def main():
    """Run the demo application."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = DemoMainWindow()
    window.show()
    
    logger.info("Demo running - documents will be restored on next launch!")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
