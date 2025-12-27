"""
Document Type Registry - Maps content types to ViewModels and Views.

Enables polymorphic document handling where different content types
(images, text, browsers) are handled by appropriate ViewModels and Views.
"""
from typing import Dict, Type, Callable, Any, Tuple, Optional
from loguru import logger

from src.ui.mvvm.document_viewmodel import DocumentViewModel


class DocumentTypeRegistry:
    """
    Maps content types to ViewModel classes and view factories.
    
    Usage:
        registry = DocumentTypeRegistry()
        
        # Register types
        registry.register(
            "image",
            ImageDocumentViewModel,
            lambda vm: ImageViewer(vm)
        )
        
        # Create document for content
        vm, view = registry.create_for_content(file_record)
    """
    
    def __init__(self):
        self._registrations: Dict[str, Tuple[Type[DocumentViewModel], Callable]] = {}
        self._type_resolvers: list = []
    
    def register(self, 
                 content_type: str,
                 viewmodel_class: Type[DocumentViewModel],
                 view_factory: Callable[[DocumentViewModel], Any]) -> None:
        """
        Register a content type mapping.
        
        Args:
            content_type: Type identifier (e.g., "image", "text", "browser")
            viewmodel_class: DocumentViewModel subclass for this type
            view_factory: Function that creates view for the ViewModel
        """
        self._registrations[content_type] = (viewmodel_class, view_factory)
        logger.debug(f"Registered document type: {content_type}")
    
    def add_type_resolver(self, resolver: Callable[[Any], Optional[str]]) -> None:
        """
        Add a function that determines content type from context.
        
        Args:
            resolver: Callable that returns content type string or None
        """
        self._type_resolvers.append(resolver)
    
    def resolve_type(self, context: Any) -> Optional[str]:
        """
        Determine content type from context.
        
        Args:
            context: Content to analyze (e.g., FileRecord)
            
        Returns:
            Content type string or None if unknown
        """
        for resolver in self._type_resolvers:
            result = resolver(context)
            if result:
                return result
        return None
    
    def create(self, content_type: str, doc_id: str, 
               locator=None) -> Tuple[DocumentViewModel, Any]:
        """
        Create ViewModel and View for a content type.
        
        Args:
            content_type: Registered type identifier
            doc_id: Document ID
            locator: ServiceLocator
            
        Returns:
            Tuple of (ViewModel, View)
        """
        if content_type not in self._registrations:
            raise ValueError(f"Unknown document type: {content_type}")
        
        vm_class, view_factory = self._registrations[content_type]
        
        viewmodel = vm_class(doc_id, locator)
        view = view_factory(viewmodel)
        
        logger.info(f"Created {content_type} document: {doc_id}")
        return viewmodel, view
    
    def create_for_content(self, context: Any, doc_id: str,
                          locator=None) -> Optional[Tuple[DocumentViewModel, Any]]:
        """
        Create ViewModel and View based on content.
        
        Args:
            context: Content to create document for
            doc_id: Document ID
            locator: ServiceLocator
            
        Returns:
            Tuple of (ViewModel, View) or None if type unknown
        """
        content_type = self.resolve_type(context)
        if not content_type:
            logger.warning(f"Could not resolve type for: {context}")
            return None
        
        return self.create(content_type, doc_id, locator)
    
    def get_registered_types(self) -> list:
        """Get list of registered content types."""
        return list(self._registrations.keys())
