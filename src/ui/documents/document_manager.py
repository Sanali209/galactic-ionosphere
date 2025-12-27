"""
Document Manager - Active document tracking and coordination.

Manages document ViewModels, tracks which document is active,
and broadcasts context changes to panels.
"""
from typing import Dict, Optional, Type, Any, Callable
from PySide6.QtCore import QObject, Signal
from loguru import logger

from src.core.base_system import BaseSystem
from src.ui.mvvm.document_viewmodel import DocumentViewModel


class _DocumentSignals(QObject):
    """Signal holder to avoid metaclass conflict."""
    active_changed = Signal(str)  # doc_id
    document_opened = Signal(str, object)  # doc_id, DocumentViewModel
    document_closed = Signal(str)  # doc_id
    context_changed = Signal(object)  # Active document's content_context


class DocumentManager(BaseSystem):
    """
    Manages document ViewModels and active state.
    
    Responsibilities:
    - Track all open documents
    - Manage active document state
    - Broadcast context to panels when active changes
    - Coordinate document lifecycle
    
    Usage:
        doc_mgr = locator.get_system(DocumentManager)
        
        # Register document
        vm = MyDocumentViewModel("doc1", locator)
        doc_mgr.register(vm)
        
        # Activate
        doc_mgr.set_active("doc1")
        
        # Listen for changes via signals property
        doc_mgr.signals.active_changed.connect(on_active_changed)
    """
    
    def __init__(self, locator, config):
        """Initialize DocumentManager."""
        super().__init__(locator, config)
        
        # Qt signals via composition
        self._signals = _DocumentSignals()
        
        self._documents: Dict[str, DocumentViewModel] = {}
        self._active_id: Optional[str] = None
        self._view_factories: Dict[Type[DocumentViewModel], Callable] = {}
    
    @property
    def signals(self) -> _DocumentSignals:
        """Get Qt signals object for UI binding."""
        return self._signals
    
    async def initialize(self):
        """Initialize the DocumentManager."""
        await super().initialize()
        logger.info("DocumentManager initialized")
    
    async def shutdown(self):
        """Shutdown and cleanup."""
        # Notify all documents of shutdown
        for doc_id in list(self._documents.keys()):
            self.close(doc_id)
        await super().shutdown()
    
    @property
    def documents(self) -> Dict[str, DocumentViewModel]:
        """Get all open documents."""
        return self._documents
    
    @property
    def active_id(self) -> Optional[str]:
        """Get active document ID."""
        return self._active_id
    
    @property
    def active(self) -> Optional[DocumentViewModel]:
        """Get active document ViewModel."""
        if self._active_id:
            return self._documents.get(self._active_id)
        return None
    
    def register(self, viewmodel: DocumentViewModel) -> None:
        """
        Register a document ViewModel.
        
        Args:
            viewmodel: DocumentViewModel to register
        """
        doc_id = viewmodel.doc_id
        
        if doc_id in self._documents:
            logger.warning(f"Document already registered: {doc_id}")
            return
        
        self._documents[doc_id] = viewmodel
        
        # Connect close request
        viewmodel.close_requested.connect(
            lambda did=doc_id: self.close(did)
        )
        
        # Set as active if first document
        if len(self._documents) == 1:
            self.set_active(doc_id)
        
        self._signals.document_opened.emit(doc_id, viewmodel)
        logger.info(f"Document registered: {doc_id}")
    
    def close(self, doc_id: str) -> bool:
        """
        Close a document.
        
        Args:
            doc_id: Document to close
            
        Returns:
            True if closed, False if prevented
        """
        if doc_id not in self._documents:
            return False
        
        viewmodel = self._documents[doc_id]
        
        # Check if can close
        if not viewmodel.can_close():
            logger.info(f"Document close prevented: {doc_id}")
            return False
        
        # Switch active if closing active
        if self._active_id == doc_id:
            remaining = [k for k in self._documents if k != doc_id]
            if remaining:
                self.set_active(remaining[0])
            else:
                self._active_id = None
                self._signals.context_changed.emit(None)
        
        del self._documents[doc_id]
        self._signals.document_closed.emit(doc_id)
        logger.info(f"Document closed: {doc_id}")
        return True
    
    def set_active(self, doc_id: str) -> None:
        """
        Set active document.
        
        Args:
            doc_id: Document to activate
        """
        if doc_id not in self._documents:
            logger.warning(f"Document not found: {doc_id}")
            return
        
        if self._active_id == doc_id:
            return
        
        # Deactivate previous
        if self._active_id and self._active_id in self._documents:
            self._documents[self._active_id].on_deactivated()
        
        # Activate new
        self._active_id = doc_id
        viewmodel = self._documents[doc_id]
        viewmodel.on_activated()
        
        # Broadcast context to panels
        self._signals.context_changed.emit(viewmodel.content_context)
        
        self._signals.active_changed.emit(doc_id)
        logger.debug(f"Active document: {doc_id}")
    
    def get(self, doc_id: str) -> Optional[DocumentViewModel]:
        """Get document ViewModel by ID."""
        return self._documents.get(doc_id)
    
    def broadcast_to_all(self, callback: Callable[[DocumentViewModel], None]) -> None:
        """
        Execute callback on all documents.
        
        Args:
            callback: Function to call with each ViewModel
        """
        for viewmodel in self._documents.values():
            callback(viewmodel)
    
    def get_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state of all documents for persistence.
        
        Returns:
            Dict mapping doc_id to state dict
        """
        return {
            doc_id: vm.get_state() 
            for doc_id, vm in self._documents.items()
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """
        Get complete session state.
        
        Returns:
            Dict with active_id and all document states
        """
        return {
            "active_id": self._active_id,
            "documents": self.get_states(),
        }
    
    # === DockingService Integration ===
    
    def connect_to_docking(self, docking_service) -> None:
        """
        Connect to DockingService for automatic synchronization.
        
        This wires up signals so that:
        - When a document tab is clicked -> set_active is called
        - When a document is closed in UI -> close is called
        
        Args:
            docking_service: The DockingService instance
        """
        self._docking_service = docking_service
        
        # Document tab activated -> set active in manager
        docking_service.document_activated.connect(self._on_docking_activated)
        
        # Document closed in UI -> remove from manager
        docking_service.document_closed.connect(self._on_docking_closed)
        
        logger.info("DocumentManager connected to DockingService")
    
    def _on_docking_activated(self, doc_id: str) -> None:
        """Handle document activation from docking."""
        if doc_id in self._documents:
            self.set_active(doc_id)
    
    def _on_docking_closed(self, doc_id: str) -> None:
        """Handle document close from docking."""
        if doc_id in self._documents:
            # Remove from our tracking (already closed in UI)
            viewmodel = self._documents.pop(doc_id, None)
            if viewmodel:
                # Switch active if needed
                if self._active_id == doc_id:
                    remaining = list(self._documents.keys())
                    if remaining:
                        self.set_active(remaining[0])
                    else:
                        self._active_id = None
                        self._signals.context_changed.emit(None)
                
                self._signals.document_closed.emit(doc_id)
    
    def create_document(self, doc_id: str, viewmodel_class: Type[DocumentViewModel] = None,
                       **kwargs) -> DocumentViewModel:
        """
        Create and register a new document ViewModel.
        
        Args:
            doc_id: Unique document ID
            viewmodel_class: DocumentViewModel subclass (optional, uses base if not provided)
            **kwargs: Additional arguments for ViewModel constructor
            
        Returns:
            The created DocumentViewModel
        """
        vm_class = viewmodel_class or DocumentViewModel
        viewmodel = vm_class(doc_id, self._locator, **kwargs)
        self.register(viewmodel)
        return viewmodel
    
    def restore_documents(self, states: Dict[str, Dict[str, Any]], 
                         viewmodel_factory: Callable[[str, Dict], DocumentViewModel]) -> None:
        """
        Restore documents from saved state.
        
        Args:
            states: Dict mapping doc_id to state dicts
            viewmodel_factory: Function(doc_id, state) -> DocumentViewModel
        """
        for doc_id, state in states.items():
            try:
                viewmodel = viewmodel_factory(doc_id, state)
                if viewmodel:
                    viewmodel.restore_state(state)
                    self.register(viewmodel)
            except Exception as e:
                logger.error(f"Failed to restore document {doc_id}: {e}")
        
        logger.info(f"Restored {len(states)} documents")
