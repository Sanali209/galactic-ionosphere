"""
Session Manager for UExplorer

Handles session persistence: saving/restoring documents, layout, and panel states.
Extracted from MainWindow for modularity.
"""
from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow
    from src.core.locator import ServiceLocator
    from src.ui.docking import DockingService


class SessionManager:
    """
    Manages session persistence for UExplorer.
    
    Responsibilities:
    - Connect SessionState to DockingService
    - Save/restore document states
    - Save/restore panel visibility
    - Handle layout persistence
    """
    
    def __init__(self, window: "QMainWindow", locator: "ServiceLocator", docking_service: "DockingService"):
        """
        Initialize session manager.
        
        Args:
            window: Main window reference (for creating documents)
            locator: ServiceLocator for accessing SessionState
            docking_service: DockingService for layout management
        """
        self.window = window
        self.locator = locator
        self.docking_service = docking_service
        self._session = None
        
    def setup_integration(self) -> bool:
        """Connect SessionState to DockingService for integrated persistence."""
        try:
            from src.ui.state import SessionState
            self._session = self.locator.get_system(SessionState)
            if self._session:
                self._session.set_docking_service(self.docking_service)
                logger.info("SessionState connected to DockingService")
                return True
        except (KeyError, ImportError) as e:
            logger.debug(f"SessionState not available: {e}")
        return False
    
    def save(self) -> bool:
        """
        Save session using Foundation's SessionState.
        
        Returns:
            True if saved successfully
        """
        try:
            from src.ui.state import SessionState
            session = self.locator.get_system(SessionState)
            if session:
                session.save()
                logger.info("Session saved via SessionState")
                return True
        except (KeyError, ImportError):
            pass
        
        # Fallback: save layout only
        try:
            if hasattr(self.window, 'save_layout'):
                self.window.save_layout()
                return True
        except RuntimeError:
            pass
        
        return False
    
    def restore(self, on_restore_document: callable = None) -> bool:
        """
        Restore session using Foundation's SessionState.
        
        Args:
            on_restore_document: Callback to restore individual documents
            
        Returns:
            True if documents were restored
        """
        try:
            from src.ui.state import SessionState
            session = self.locator.get_system(SessionState)
            if not session:
                return False
            
            # Get saved document states
            doc_states = session.get_document_states()
            if not doc_states:
                logger.info("No documents to restore from session")
                return False
            
            logger.info(f"Restoring session: {len(doc_states)} documents")
            restored = 0
            
            for doc_id, doc_state in doc_states.items():
                # Skip closed documents
                if doc_state.get("is_closed", False):
                    logger.debug(f"Skipping closed document: {doc_id}")
                    continue
                    
                try:
                    if on_restore_document:
                        on_restore_document(doc_id, doc_state)
                        restored += 1
                except Exception as e:
                    logger.error(f"Failed to restore {doc_id}: {e}")
            
            # Restore layout
            self._restore_layout(session, restored)
            
            logger.info(f"Session restored: {restored} documents")
            return restored > 0
            
        except (KeyError, ImportError) as e:
            logger.debug(f"SessionState not available: {e}")
            return False
    
    def _restore_layout(self, session, restored_count: int):
        """Restore docking layout from session."""
        docking_state = session.get("docking", {})
        layout = docking_state.get("layout_bytes") if isinstance(docking_state, dict) else None
        
        if layout and restored_count > 0:
            try:
                layout_bytes = bytes.fromhex(layout)
                self.docking_service.restore_layout(layout_bytes)
                logger.info("Restored docking layout from session")
            except Exception as e:
                logger.warning(f"Failed to restore layout: {e}")
                # Fallback to panel visibility restoration
                panel_states = docking_state.get("panels", {})
                self._restore_panel_visibility(panel_states)
        else:
            # No layout bytes, try legacy panel restoration
            panel_states = docking_state.get("panels", {}) if isinstance(docking_state, dict) else {}
            self._restore_panel_visibility(panel_states)
    
    def _restore_panel_visibility(self, panel_states: Dict[str, Any]):
        """Restore panel visibility as fallback when layout_bytes fails."""
        if not panel_states:
            return
            
        for panel_id, panel_state in panel_states.items():
            try:
                is_visible = panel_state.get("is_visible", True)
                if is_visible:
                    self.docking_service.show_panel(panel_id)
                else:
                    self.docking_service.hide_panel(panel_id)
            except Exception as e:
                logger.debug(f"Failed to restore panel {panel_id}: {e}")
        logger.info("Restored panel visibility from session")


def restore_browser_document(
    window: "QMainWindow",
    locator: "ServiceLocator", 
    docking_service: "DockingService",
    doc_id: str, 
    doc_state: dict
) -> None:
    """
    Restore a browser document from saved state.
    
    Args:
        window: MainWindow reference
        locator: ServiceLocator
        docking_service: DockingService
        doc_id: Document ID
        doc_state: Saved state dict
    """
    from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
    
    title = doc_state.get("title", "Files")
    custom_state = doc_state.get("custom_state", {})
    
    # Register with DocumentManager FIRST
    vm = None
    if hasattr(window, 'document_manager'):
        vm = window.document_manager.create_document(doc_id)
    
    # Create document with viewmodel
    doc = FileBrowserDocument(locator, viewmodel=vm, title=title, parent=None)
    
    # Restore viewmodel state
    if custom_state:
        doc.set_state(custom_state)
    
    docking_service.add_document(doc_id, doc, title, area="center", closable=True)
    
    # Connect selection to metadata panel
    if hasattr(window, 'on_selection_changed'):
        doc.selection_changed.connect(window.on_selection_changed)
    
    logger.debug(f"Restored document: {doc_id} - {title}")


def open_browser_for_directory(
    window: "QMainWindow",
    locator: "ServiceLocator",
    docking_service: "DockingService",
    directory_id: str
) -> str:
    """
    Open a new browser tab and navigate to directory.
    
    Args:
        window: MainWindow reference
        locator: ServiceLocator
        docking_service: DockingService
        directory_id: Directory ID to navigate to
        
    Returns:
        Created document ID
    """
    from uexplorer_src.ui.documents.file_browser_document import FileBrowserDocument
    import uuid
    
    doc_id = f"browser_{uuid.uuid4().hex[:8]}"
    
    # Create ViewModel via DocumentManager
    vm = None
    if hasattr(window, 'document_manager'):
        vm = window.document_manager.create_document(doc_id)
        
    doc = FileBrowserDocument(locator, viewmodel=vm, title="Files", parent=None)
    
    docking_service.add_document(doc_id, doc, "Files", area="center", closable=True)
    doc.browse_directory(directory_id)
    
    return doc_id
