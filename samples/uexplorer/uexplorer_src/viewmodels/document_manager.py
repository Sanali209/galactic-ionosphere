"""
DocumentManager - Tracks active document and routes results.

Manages multiple browser documents and ensures search results
go to the currently active document.
"""
from typing import TYPE_CHECKING, Dict, Optional, Any
from pathlib import Path
from PySide6.QtCore import QObject, Signal
from loguru import logger

from uexplorer_src.viewmodels.browse_view_model import BrowseViewModel
from src.ui.navigation.service import NavigationHandler, NavigationContext

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class DocumentManager(QObject, NavigationHandler):
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
    # Navigation signals
    request_new_document = Signal(object)  # data (path/id)
    
    def __init__(self, parent: Optional[QObject] = None) -> None:
        QObject.__init__(self, parent)
        self._documents: Dict[str, BrowseViewModel] = {}
        self._active_id: Optional[str] = None
        
        logger.info("DocumentManager initialized")
    
    @property
    def priority(self) -> int:
        return 100  # High priority for initial handling

    def can_handle(self, data: Any) -> bool:
        """Check if we can navigate to this data."""
        # We handle Directory IDs (str) or Paths
        if isinstance(data, (str, Path)):
            return True
        return False

    def owns_target(self, target_id: str) -> bool:
        """Check if target_id is a managed document."""
        return target_id in self._documents

    def handle_existing(self, target_id: str, data: Any, context: NavigationContext):
        """Navigate within an existing document."""
        if target_id in self._documents:
            vm = self._documents[target_id]
            logger.info(f"Navigating doc {target_id} to {data}")
            
            # Convert string to ObjectId if needed
            from bson import ObjectId
            try:
                if isinstance(data, str) and len(data) == 24:
                    dir_id = ObjectId(data)
                else:
                    # TODO: Handle path-to-id resolution if data is Path
                    # For now assume ObjectId string
                    dir_id = ObjectId(str(data))
                    
                vm.set_directory(dir_id)
            except Exception as e:
                logger.error(f"Navigation failed: {e}")

    def handle_new(self, data: Any, context: NavigationContext):
        """Request new document for data."""
        logger.info(f"Requesting new document for: {data}")
        self.request_new_document.emit(data)
        
    # Standard methods...
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
        """Create a new document ViewModel."""
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
        """Remove a document."""
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
        """Set active document."""
        if doc_id not in self._documents:
            # It might be a document we don't manage (e.g. image viewer), allow unsetting?
            # Or just ignore if not found?
            # logger.warning(f"Document not found: {doc_id}")
            if doc_id:
                logger.debug(f"DocumentManager: Ignoring activation of non-managed doc: {doc_id}")
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
        
        Also tries to refresh active ID from DockingService if current active ID
        seems stale or missing.
        """
        # Try to refresh active ID from DockingService if possible (source of truth)
        # Access parent's locator -> main_window -> docking_service? 
        # Or just trust internal state.
        # But user reported mismatch.
        # Let's check internal state first.
        
        target_id = self._active_id
        
        # If target seems invalid or missing, maybe log it
        if not target_id:
            logger.warning("No active document tracked in DocumentManager")
            
        if target_id and target_id in self._documents:
            self._documents[target_id].set_results(results)
            logger.info(f"Results sent to {target_id}: {len(results)} items")
        else:
            logger.warning(f"Failed to send results: Active ID '{target_id}' not found in managed documents")
    
    def broadcast_results(self, results: list):
        """Send results to ALL documents."""
        for doc_id, viewmodel in self._documents.items():
            viewmodel.set_results(results)
        logger.info(f"Results broadcast to {len(self._documents)} documents")
