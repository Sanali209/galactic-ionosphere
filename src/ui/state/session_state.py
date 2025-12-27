"""
Session State - Persists and restores document/panel state.

Enables applications to restore their state on startup, including
which documents were open, their positions, and active state.

Integrates with:
- DockingService: Layout, panel visibility, perspectives
- DocumentManager: Open documents and their custom state
"""
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
from loguru import logger

from src.core.base_system import BaseSystem


class SessionState(BaseSystem):
    """
    Persists and restores UI session state.
    
    Saves:
    - Open documents and their states
    - Active document ID
    - Panel visibility and auto-hide states
    - Docking layout (positions, sizes)
    - Perspectives (named layouts)
    
    Usage:
        # Get session service
        session = locator.get_system(SessionState)
        
        # Set docking service reference
        session.set_docking_service(docking_service)
        
        # On shutdown (called automatically)
        session.save()
        
        # On startup
        session.restore()
        
        # Or restore with document factory
        session.restore_with_factory(create_document_func)
    """
    
    def __init__(self, locator, config):
        """Initialize SessionState."""
        super().__init__(locator, config)
        
        # Get save path from config or use default
        self._save_path = Path(getattr(config.data, 'session_file', 'session.json'))
        self._state: Dict[str, Any] = {}
        self._docking_service = None
    
    async def initialize(self):
        """Initialize and load previous session."""
        await super().initialize()
        self._load_from_disk()
        logger.info(f"SessionState initialized (file: {self._save_path})")
    
    async def shutdown(self):
        """Save session on shutdown."""
        self.save()
        await super().shutdown()
    
    def set_docking_service(self, docking_service) -> None:
        """
        Set reference to the DockingService.
        
        Args:
            docking_service: The application's DockingService instance
        """
        self._docking_service = docking_service
    
    def _get_docking_service(self):
        """Get docking service from stored reference or locator."""
        if self._docking_service:
            return self._docking_service
        
        # Try to get from locator (common pattern)
        try:
            if hasattr(self.locator, 'docking_service'):
                return self.locator.docking_service
        except Exception:
            pass
        
        return None
    
    def _load_from_disk(self) -> None:
        """Load session state from disk."""
        if self._save_path.exists():
            try:
                with open(self._save_path, 'r') as f:
                    self._state = json.load(f)
                logger.info(f"Loaded session state: {len(self._state)} keys")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")
                self._state = {}
    
    def save(self) -> None:
        """Save session state to disk."""
        try:
            # Collect state from managers
            self._collect_state()
            
            # Ensure parent directory exists
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._save_path, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
            
            logger.info(f"Session saved to {self._save_path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _collect_state(self) -> None:
        """Collect state from all managers."""
        # Docking state (includes layout, documents, panels, perspectives)
        docking = self._get_docking_service()
        if docking:
            try:
                self._state['docking'] = docking.get_complete_state()
                logger.debug(f"Collected docking state: {len(self._state['docking'])} keys")
            except Exception as e:
                logger.warning(f"Failed to collect docking state: {e}")
        
        # Document Manager state (ViewModels)
        try:
            from src.ui.documents.document_manager import DocumentManager
            doc_mgr = self.locator.get_system(DocumentManager)
            if doc_mgr:
                self._state['document_manager'] = doc_mgr.get_session_state()
        except (KeyError, ImportError) as e:
            logger.debug(f"DocumentManager not available: {e}")
    
    def restore(self) -> None:
        """
        Restore session state.
        
        Call this after DockingService is initialized but BEFORE
        adding documents. The layout will be restored, then you
        should restore documents using the saved state.
        """
        if not self._state:
            logger.info("No session to restore")
            return
        
        # Restore docking layout
        docking_state = self._state.get('docking', {})
        if 'layout_bytes' in docking_state and docking_state['layout_bytes']:
            self._restore_layout(docking_state['layout_bytes'])
        
        logger.info("Session restored (layout)")
    
    def restore_with_factory(self, document_factory: Callable[[str, Dict], Any]) -> None:
        """
        Restore session including recreating documents.
        
        Args:
            document_factory: Function(doc_id, state) -> widget
                Called for each saved document to recreate it
        """
        if not self._state:
            logger.info("No session to restore")
            return
        
        docking = self._get_docking_service()
        docking_state = self._state.get('docking', {})
        
        # Restore documents first
        doc_states = docking_state.get('documents', {})
        for doc_id, state in doc_states.items():
            try:
                widget = document_factory(doc_id, state.get('custom_state', {}))
                if widget and docking:
                    docking.add_document(
                        doc_id=doc_id,
                        widget=widget,
                        title=state.get('title', 'Untitled')
                    )
            except Exception as e:
                logger.error(f"Failed to restore document {doc_id}: {e}")
        
        # Restore layout (positions, sizes)
        if 'layout_bytes' in docking_state and docking_state['layout_bytes']:
            self._restore_layout(docking_state['layout_bytes'])
        
        # Activate previously active document
        active_doc = docking_state.get('active_document')
        if active_doc and docking:
            docking.activate_document(active_doc)
        
        logger.info(f"Session restored: {len(doc_states)} documents")
    
    def _restore_layout(self, layout_hex: str) -> None:
        """Restore docking layout from hex string."""
        try:
            docking = self._get_docking_service()
            if docking:
                layout_bytes = bytes.fromhex(layout_hex)
                docking.restore_layout(layout_bytes)
                logger.debug("Docking layout restored")
        except Exception as e:
            logger.warning(f"Failed to restore layout: {e}")
    
    # === State Accessors ===
    
    def get_document_states(self) -> Dict[str, Dict[str, Any]]:
        """Get saved document states for restoration."""
        docking_state = self._state.get('docking', {})
        return docking_state.get('documents', {})
    
    def get_panel_states(self) -> Dict[str, Dict[str, Any]]:
        """Get saved panel states."""
        docking_state = self._state.get('docking', {})
        return docking_state.get('panels', {})
    
    def get_active_document_id(self) -> Optional[str]:
        """Get previously active document ID."""
        docking_state = self._state.get('docking', {})
        return docking_state.get('active_document')
    
    def get_perspectives(self) -> List[str]:
        """Get saved perspective names."""
        docking_state = self._state.get('docking', {})
        return docking_state.get('perspectives', [])
    
    # === Custom State ===
    
    def set(self, key: str, value: Any) -> None:
        """Set a custom session value."""
        self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a custom session value."""
        return self._state.get(key, default)
    
    # === Panel State Helpers ===
    
    def restore_panel_states(self) -> None:
        """
        Restore panel visibility and auto-hide states.
        
        Call after panels are added to DockingService.
        """
        docking = self._get_docking_service()
        if not docking:
            return
        
        panel_states = self.get_panel_states()
        for panel_id, state in panel_states.items():
            try:
                # Restore visibility
                if state.get('is_visible', True):
                    docking.show_panel(panel_id)
                else:
                    docking.hide_panel(panel_id)
                
                # Restore auto-hide
                if state.get('is_auto_hide'):
                    sidebar = state.get('sidebar', 'right')
                    docking.set_panel_auto_hide(panel_id, True, sidebar)
                    
            except Exception as e:
                logger.debug(f"Could not restore panel {panel_id}: {e}")
