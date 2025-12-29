"""
Navigation Service - Intelligent routing for user actions.

This service orchestrates navigation requests, deciding where to open
content based on context, active focus, and handler capabilities.
"""
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal
from loguru import logger

from src.core.base_system import BaseSystem


@dataclass
class NavigationContext:
    """Context for a navigation request."""
    source_id: str = ""       # ID of the widget initiating the request
    target_id: Optional[str] = None  # Specific target ID if known
    new_window: bool = False  # Force new window/tab
    metadata: Dict[str, Any] = None


class NavigationService(BaseSystem):
    """
    Centralized navigation orchestrator.
    
    Routes data (like file records, directory paths) to the most appropriate
    UI component (Document or Panel).
    
    Usage:
        nav = locator.get_system(NavigationService)
        nav.navigate(file_record, source_id="browser_1")
    """
    
    depends_on = []  # Independent UI service
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._handlers: List['NavigationHandler'] = []
        self._docking_service_cached = None
        
    async def initialize(self):
        """Initialize service and resolve dependencies."""
        await super().initialize()
        
        # DockingService will be lazily resolved via @property
        logger.info("NavigationService initialized")

    async def shutdown(self):
        """Shutdown service."""
        self._handlers.clear()
        self._docking_service = None
        logger.info("NavigationService shutdown")

    def register_handler(self, handler: 'NavigationHandler'):
        """Register a navigation handler capability."""
        if handler not in self._handlers:
            self._handlers.append(handler)
            # Sort by priority (higher first)
            self._handlers.sort(key=lambda h: h.priority, reverse=True)
            logger.debug(f"Registered navigation handler: {handler}")

    def navigate(self, data: Any, context: Optional[NavigationContext] = None) -> bool:
        """
        Route data to an appropriate handler.
        
        Args:
            data: The object/data to navigate to (FileRecord, Path, etc.)
            context: Navigation context (source, preferences)
            
        Returns:
            True if handled, False otherwise.
        """
        if context is None:
            context = NavigationContext()
            
        logger.info(f"Navigating: {data} (Source: {context.source_id})")
        
        # 1. Check for explicit target
        if context.target_id:
            if self._try_target(context.target_id, data, context):
                return True
        
        # 2. Get active active target (if compatible)
        active_id = self._get_active_target_id()
        if active_id and active_id != context.source_id:
             if self._try_target(active_id, data, context):
                return True
        
        # 3. Find ANY compatible open document
        # Iterate all documents to see if one can be reused (e.g. browsing different folder in same tab)
        if self.docking_service:
            # Get all document IDs
            doc_ids = self.docking_service.documents.keys()
            for doc_id in doc_ids:
                # Ask handlers if they own this document
                for handler in self._handlers:
                    if handler.owns_target(doc_id):
                        if handler.can_handle(data):
                             logger.info(f"Reuse compatible document: {doc_id}")
                             handler.handle_existing(doc_id, data, context)
                             return True

        # 4. Create NEW document using best handler
        for handler in self._handlers:
            if handler.can_handle(data):
                logger.info(f"Creating new view via handler: {handler}")
                handler.handle_new(data, context)
                return True
                
        logger.warning(f"No handler found for data: {data}")
        return False

    @property
    def docking_service(self):
        """Lazy access to DockingService."""
        if not self._docking_service_cached:
            from src.ui.docking import DockingService
            try:
                self._docking_service_cached = self.locator.get_system(DockingService)
                logger.info("NavigationService lazily connected to DockingService")
            except KeyError:
                return None
        return self._docking_service_cached

    def _get_active_target_id(self) -> Optional[str]:
        """Get the ID of the currently focused document/panel."""
        if not self.docking_service:
            return None
        return self.docking_service.get_active_document_id()

    def _try_target(self, target_id: str, data: Any, context: NavigationContext) -> bool:
        """Attempt to handle navigation in a specific target."""
        # Find handler that claims this target_id
        # In a real system, the Document/Widget itself might implement an interface.
        # Here we ask registered handlers if they "own" this target_id.
        
        for handler in self._handlers:
            if handler.owns_target(target_id):
                if handler.can_handle(data):
                    logger.info(f"Routing to existing target: {target_id}")
                    handler.handle_existing(target_id, data, context)
                    return True
        return False


class NavigationHandler:
    """Base class/Interface for navigation capabilities."""
    
    @property
    def priority(self) -> int:
        return 0
        
    def can_handle(self, data: Any) -> bool:
        """Return True if this handler can process the data."""
        return False
        
    def owns_target(self, target_id: str) -> bool:
        """Return True if check target_id belongs to this handler."""
        return False
        
    def handle_existing(self, target_id: str, data: Any, context: NavigationContext):
        """Navigate within an existing view."""
        pass
        
    def handle_new(self, data: Any, context: NavigationContext):
        """Create a new view for the data."""
        pass
