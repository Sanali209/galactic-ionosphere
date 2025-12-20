"""
Container widget for documents in a split area.
Manages tab bar and document widgets.
"""
from typing import List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Signal
from loguru import logger
from .document_view import DocumentView
from .docking.draggable_tab_bar import DraggableTabBar


class SplitContainer(QWidget):
    """
    Holds multiple documents (tabs) in one split area.
    
    With drag & drop support via DraggableTabBar.
    """
    # Signals
    document_added = Signal(object)  # DocumentView
    document_removed = Signal(object)  # DocumentView
    document_activated = Signal(object)  # DocumentView
    all_closed = Signal()  # Emitted when last tab closes
    
    # Drag & drop signals (from tab bar)
    document_drag_started = Signal(str, str)  # doc_id, container_id
    document_dropped = Signal(str, int)  # document_id, insert_index
    drag_ended = Signal()  # Drag operation completed
    
    def __init__(self, node_id: str, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.documents: List[DocumentView] = []
        
        # UI Setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget(self)
        
        # Use draggable tab bar
        self.draggable_tab_bar = DraggableTabBar()
        self.tab_widget.setTabBar(self.draggable_tab_bar)
        
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_widget.currentChanged.connect(self._on_current_changed)
        
        layout.addWidget(self.tab_widget)
        
        # Forward signals from tab bar
        self.draggable_tab_bar.tab_drag_started.connect(
            lambda doc_id: self.document_drag_started.emit(doc_id, self.node_id)
        )
        self.draggable_tab_bar.tab_dropped.connect(self.document_dropped)
        self.draggable_tab_bar.drag_ended.connect(
            lambda: self.drag_ended.emit()
        )
        
        logger.debug(f"SplitContainer created with DraggableTabBar: {node_id}")
    
    def add_document(self, doc: "DocumentView", title: str):
        """Add a document to this container."""
        # Store document
        self.documents.append(doc)
        
        # Create content widget
        content = doc.create_content() if hasattr(doc, 'create_content') else doc
        
        # Add tab
        idx = self.tab_widget.addTab(content, title)
        self.draggable_tab_bar.setTabData(idx, doc.id if hasattr(doc, 'id') else str(id(doc)))
        self.tab_widget.setCurrentIndex(idx)
        
        # Connect signals
        if hasattr(doc, 'content_changed'):
            doc.content_changed.connect(lambda: self._update_tab_title(doc))
        
        # Emit signal
        self.document_added.emit(doc)
        
        doc_id = doc.id if hasattr(doc, 'id') else str(id(doc))
        logger.info(f"Document added to {self.node_id}: {title} (ID: {doc_id})")
    
    def remove_document(self, document_id: str) -> Optional["DocumentView"]:
        """Remove a document by ID and return it."""
        for i, doc in enumerate(self.documents):
            if doc.id == document_id:
                # Remove from list
                removed_doc = self.documents.pop(i)
                
                # Find and remove tab
                for tab_idx in range(self.tab_widget.count()):
                    if self.draggable_tab_bar.tabData(tab_idx) == document_id:
                        self.tab_widget.removeTab(tab_idx)
                        break
                
                # Emit signals
                self.document_removed.emit(removed_doc)
                if len(self.documents) == 0:
                    self.all_closed.emit()
                
                logger.info(f"Document removed from {self.node_id}: {document_id}")
                return removed_doc
        
        logger.warning(f"Document not found for removal: {document_id}")
        return None
    
    def get_document(self, document_id: str) -> Optional["DocumentView"]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None
    
    def reorder_document(self, document_id: str, new_index: int):
        """Move a document to a specific tab position."""
        # Find current index in self.documents
        old_index_in_list = None
        for i, doc in enumerate(self.documents):
            if doc.id == document_id:
                old_index_in_list = i
                break
        
        if old_index_in_list is None:
            logger.warning(f"Cannot reorder - document not found in list: {document_id}")
            return
        
        # Find current index in tab widget
        old_index_in_tabs = -1
        for tab_idx in range(self.tab_widget.count()):
            if self.tab_widget.tabData(tab_idx) == document_id:
                old_index_in_tabs = tab_idx
                break
        
        if old_index_in_tabs == -1:
            logger.warning(f"Cannot reorder - document tab not found: {document_id}")
            return
        
        # Reorder in list
        doc = self.documents.pop(old_index_in_list)
        self.documents.insert(new_index, doc)
        
        # Reorder tab
        # Get tab properties
        title = self.tab_widget.tabText(old_index_in_tabs)
        widget = self.tab_widget.widget(old_index_in_tabs)
        
        # Remove and re-insert
        self.tab_widget.removeTab(old_index_in_tabs)
        self.tab_widget.insertTab(new_index, widget, title)
        self.tab_widget.setTabData(new_index, document_id)
        self.tab_widget.setCurrentIndex(new_index)
        
        logger.info(f"Document reordered in {self.node_id}: {document_id} to index {new_index}")
    
    def remove_document_by_index(self, index: int) -> Optional[DocumentView]:
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
