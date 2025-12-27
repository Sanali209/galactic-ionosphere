"""
Document management system.

Provides:
- BaseDocumentWidget: Base widget for document views
- DocumentManager: Tracks open documents and active state
- DocumentTypeRegistry: Maps content types to ViewModels/Views
- SplitManager: Manages document splitting layouts
"""
from src.ui.documents.base_document import BaseDocumentWidget
from src.ui.documents.document_manager import DocumentManager
from src.ui.documents.registry import DocumentTypeRegistry

__all__ = [
    'BaseDocumentWidget',
    'DocumentManager', 
    'DocumentTypeRegistry',
]
