"""
DocumentManager - Tracks active document and routes results.

Manages multiple browser documents and ensures search results
go to the currently active document.
"""
from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal
from loguru import logger

from uexplorer_src.viewmodels.browse_view_model import BrowseViewModel


class DocumentManager(QObject):
    """
    Manages multiple BrowseViewModels and tracks active document.
    
    Signals:
        active_changed: Active document changed
        document_added: New document created
        document_removed: Document closed
    """
    
    active_changed = Signal(str)  # doc_id
    document_added = Signal(str, object)  # doc_id, BrowseViewModel
    document_removed = Signal(str)  # doc_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._documents: Dict[str, BrowseViewModel] = {}
        self._active_id: Optional[str] = None
        
        logger.info("DocumentManager initialized")
    
    @property
    def active_id(self) -> Optional[str]:
        """Get active document ID."""
        return self._active_id
    
    @property
    def active_viewmodel(self) -> Optional[BrowseViewModel]:
        """Get active document's ViewModel."""
        if self._active_id and self._active_id in self._documents:
            return self._documents[self._active_id]
        return None
    
    @property
    def documents(self) -> Dict[str, BrowseViewModel]:
        """Get all documents."""
        return self._documents
    
    def create_document(self, doc_id: str) -> BrowseViewModel:
        """
        Create a new document ViewModel.
        
        Args:
            doc_id: Unique document identifier
            
        Returns:
            New BrowseViewModel instance
        """
        if doc_id in self._documents:
            logger.warning(f"Document already exists: {doc_id}")
            return self._documents[doc_id]
        
        viewmodel = BrowseViewModel(doc_id, parent=self)
        self._documents[doc_id] = viewmodel
        
        # Set as active if first document
        if len(self._documents) == 1:
            self.set_active(doc_id)
        
        self.document_added.emit(doc_id, viewmodel)
        logger.info(f"Document created: {doc_id}")
        return viewmodel
    
    def remove_document(self, doc_id: str):
        """
        Remove a document.
        
        Args:
            doc_id: Document to remove
        """
        if doc_id not in self._documents:
            return
        
        # Switch active if removing active
        if self._active_id == doc_id:
            remaining = [k for k in self._documents if k != doc_id]
            if remaining:
                self.set_active(remaining[0])
            else:
                self._active_id = None
        
        del self._documents[doc_id]
        self.document_removed.emit(doc_id)
        logger.info(f"Document removed: {doc_id}")
    
    def set_active(self, doc_id: str):
        """
        Set active document.
        
        Args:
            doc_id: Document to activate
        """
        if doc_id not in self._documents:
            logger.warning(f"Document not found: {doc_id}")
            return
        
        if self._active_id != doc_id:
            self._active_id = doc_id
            self.active_changed.emit(doc_id)
            logger.debug(f"Active document: {doc_id}")
    
    def get_viewmodel(self, doc_id: str) -> Optional[BrowseViewModel]:
        """Get ViewModel by document ID."""
        return self._documents.get(doc_id)
    
    def send_results_to_active(self, results: list):
        """
        Send search results to active document.
        
        Args:
            results: List of FileRecord objects
        """
        if self._active_id and self._active_id in self._documents:
            self._documents[self._active_id].set_results(results)
            logger.info(f"Results sent to {self._active_id}: {len(results)} items")
        else:
            logger.warning("No active document to receive results")
    
    def broadcast_results(self, results: list):
        """
        Send results to ALL documents (for global filters).
        
        Args:
            results: List of FileRecord objects
        """
        for doc_id, viewmodel in self._documents.items():
            viewmodel.set_results(results)
        logger.info(f"Results broadcast to {len(self._documents)} documents")
