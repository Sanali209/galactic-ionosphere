"""
Container widget for documents in a split area.
Manages tab bar and document widgets.
"""
from typing import List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTabBar
from PySide6.QtCore import Signal, Qt
from loguru import logger
from .document_view import DocumentView

class DocumentTabBar(QTabBar):
    """
    Custom tab bar with drag-and-drop support for reordering/moving tabs.
    """
    tab_drag_started = Signal(int, object)  # index, DocumentView
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setExpanding(False)
    
    def mousePressEvent(self, event):
        """Initiate drag on mouse press."""
        if event.button() == Qt.LeftButton:
            index = self.tabAt(event.pos())
            if index >= 0:
                # Emit signal for potential cross-split drag
                parent_container = self.parent()
                if hasattr(parent_container, 'get_document'):
                    doc = parent_container.get_document(index)
                    self.tab_drag_started.emit(index, doc)
        super().mousePressEvent(event)

class SplitContainer(QWidget

):
    """
    Holds multiple documents (tabs) in one split area.
    """
    # Signals
    document_added = Signal(object)  # DocumentView
    document_removed = Signal(object)
    document_activated = Signal(object)
    all_closed = Signal()  # Emitted when last tab closes
    
    def __init__(self, node_id: str, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.documents: List[DocumentView] = []
        
        # UI Setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabBar(DocumentTabBar())
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_widget.currentChanged.connect(self._on_current_changed)
        
        layout.addWidget(self.tab_widget)
        
        logger.debug(f"SplitContainer created: {node_id}")
    
    def add_document(self, doc: DocumentView, title: str = None):
        """Add a document to this container."""
        if title is None:
            title = doc.viewmodel.title
        
        self.documents.append(doc)
        index = self.tab_widget.addTab(doc, title)
        self.tab_widget.setCurrentIndex(index)
        
        # Connect signals
        doc.content_changed.connect(lambda: self._update_tab_title(doc))
        
        self.document_added.emit(doc)
        logger.info(f"Document added to {self.node_id}: {title}")
    
    def remove_document(self, index: int) -> Optional[DocumentView]:
        """Remove document at index. Returns the removed document."""
        if 0 <= index < len(self.documents):
            doc = self.documents.pop(index)
            self.tab_widget.removeTab(index)
            self.document_removed.emit(doc)
            
            # Check if container is now empty
            if len(self.documents) == 0:
                self.all_closed.emit()
                logger.info(f"SplitContainer {self.node_id} is now empty")
            
            return doc
        return None
    
    def get_document(self, index: int) -> Optional[DocumentView]:
        """Get document at index."""
        if 0 <= index < len(self.documents):
            return self.documents[index]
        return None
    
    def get_active_document(self) -> Optional[DocumentView]:
        """Get currently active/visible document."""
        index = self.tab_widget.currentIndex()
        return self.get_document(index)
    
    def document_count(self) -> int:
        """Return number of documents in this container."""
        return len(self.documents)
    
    def _on_tab_close_requested(self, index: int):
        """Handle tab close button click."""
        doc = self.get_document(index)
        if doc and doc.can_close():
            self.remove_document(index)
    
    def _on_current_changed(self, index: int):
        """Handle tab selection change."""
        doc = self.get_document(index)
        if doc:
            self.document_activated.emit(doc)
    
    def _update_tab_title(self, doc: DocumentView):
        """Update tab title when document modified state changes."""
        try:
            index = self.documents.index(doc)
            self.tab_widget.setTabText(index, doc.viewmodel.title)
        except ValueError:
            pass  # Document not in this container
