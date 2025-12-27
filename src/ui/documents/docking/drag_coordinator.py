"""
Drag Coordinator

Central manager for drag & drop operations across all containers.
Coordinates overlays, previews, and drop execution.
"""
from typing import Optional, Dict
from PySide6.QtCore import QObject, Qt, QRect, QTimer
from PySide6.QtWidgets import QWidget, QApplication
from loguru import logger

from .drop_zone_overlay import DropZone, DropZoneOverlay
from .drop_preview import DropPreview
from ..split_manager import SplitOrientation


class DragCoordinator(QObject):
    """
    Coordinates drag & drop across all split containers.
    
    Responsibilities:
    - Show drop zone overlays on all containers during drag
    - Display preview of drop result
    - Execute drop operations (move document or create split)
    - Clean up overlays when drag ends
    
    Works with SplitManager to access containers and perform operations.
    """
    
    def __init__(self, split_manager):
        """
        Initialize drag coordinator.
        
        Args:
            split_manager: SplitManager instance
        """
        super().__init__()
        self.split_manager = split_manager
        self.active_drag: Optional[Dict] = None
        self.overlays: Dict[str, DropZoneOverlay] = {}
        self.previews: Dict[str, DropPreview] = {}
        
        logger.info("DragCoordinator initialized")
    
    def start_drag(self, document_id: str, source_container_id: str):
        """
        Start drag operation - show overlays on all containers.
        
        Args:
            document_id: ID of document being dragged
            source_container_id: ID of source container
        """
        if self.active_drag:
            logger.warning("Drag already in progress, ending previous drag")
            self.end_drag()
        
        self.active_drag = {
            'document_id': document_id,
            'source': source_container_id
        }
        
        logger.info(f"Starting drag: document={document_id}, source={source_container_id}")
        
        # Show drop zone overlays on all containers
        for node in self.split_manager.get_all_containers():
            if not node.container_widget:
                continue
            
            container_widget = node.container_widget
            
            # Always show all 5 zones - user can split even from same container
            # CENTER = reorder tabs, Edges = create split
            show_split_zones = True
            
            # Create overlay as CHILD widget of container
            overlay = DropZoneOverlay(container_widget, show_split_zones=show_split_zones)
            
            overlay.zone_entered.connect(
                lambda z, nid=node.id: self._on_zone_entered(nid, z)
            )
            overlay.zone_exited.connect(
                lambda nid=node.id: self._on_zone_exited(nid)
            )
            overlay.drop_requested.connect(
                lambda z, nid=node.id: self._execute_drop(nid, z)
            )
            
            # Show overlay - it's already positioned over parent
            overlay.show()
            overlay.raise_()  # Bring to front of children
            
            self.overlays[node.id] = overlay
            logger.debug(f"Overlay shown as child of container: {node.id}")
        
        logger.info(f"Drag started with {len(self.overlays)} overlays")
    
    def end_drag(self):
        """Clean up after drag operation ends."""
        # Guard against multiple calls with a flag
        if hasattr(self, '_cleaning_up') and self._cleaning_up:
            return  # Already cleaning up
        
        self._cleaning_up = True
        logger.info("Ending drag")
        
        # Hide and delete all overlays
        for overlay in list(self.overlays.values()):  # Use list() to avoid dict change during iteration
            overlay.hide()
            overlay.deleteLater()
        
        self.overlays.clear()
        self.active_drag = None
        
        # Close all previews
        self._hide_all_previews()
        
        self._cleaning_up = False
        logger.info("Drag ended, cleanup complete")
    
    def _on_zone_entered(self, container_id: str, zone: DropZone):
        """
        Show preview when cursor enters a drop zone.
        
        Args:
            container_id: Container ID where zone was entered
            zone: Drop zone that was entered
        """
        logger.debug(f"Zone entered: container={container_id}, zone={zone.name}")
        
        # Hide all previous previews
        self._hide_all_previews()
        
        # Calculate preview rectangle
        preview_rect = self._calculate_preview_rect(container_id, zone)
        
        # Get container widget
        node = self.split_manager.get_node(container_id)
        if not node or not node.container_widget:
            return
        
        # Create and show preview
        preview = DropPreview(zone, node.container_widget)
        preview.setGeometry(preview_rect)
        preview.show()
        preview.raise_()
        
        self.previews[container_id] = preview
        logger.debug(f"Preview shown: {preview_rect}")
    
    def _on_zone_exited(self, container_id: str):
        """
        Hide preview when cursor exits zone.
        
        Args:
            container_id: Container ID
        """
        if container_id in self.previews:
            self.previews[container_id].close()
            self.previews[container_id].deleteLater()
            del self.previews[container_id]
            logger.debug(f"Preview hidden for: {container_id}")
    
    def _execute_drop(self, target_container_id: str, zone: DropZone):
        """
        Execute drop operation.
        
        Args:
            target_container_id: Container where drop occurred
            zone: Drop zone
        """
        if not self.active_drag:
            logger.warning("No active drag")
            return
        
        doc_id = self.active_drag['document_id']
        source_id = self.active_drag['source']
        
        logger.info(f"Executing drop: doc={doc_id}, target={target_container_id}, zone={zone.name}")
        
        # Get source container and document BEFORE any splitting
        source_node = self.split_manager.get_node(source_id)
        if not source_node or not source_node.container_widget:
            logger.error(f"Source container not found: {source_id}")
            self.end_drag()
            return
        
        source_container = source_node.container_widget
        
        # Remove document from source FIRST (before split destroys container)
        document = source_container.remove_document(doc_id)
        if not document:
            logger.error(f"Document not found in source: {doc_id}")
            self.end_drag()
            return
        
        if zone == DropZone.CENTER:
            # Simple case: just add to target (same container or different)
            target_node = self.split_manager.get_node(target_container_id)
            if target_node and target_node.container_widget:
                target_node.container_widget.add_document(
                    document, 
                    document.title if hasattr(document, 'title') else "Document"
                )
                logger.info(f"✓ Document moved to existing container: {doc_id}")
        else:
            # Edge zone: create split
            orientation = SplitOrientation.HORIZONTAL if zone in [DropZone.LEFT, DropZone.RIGHT] else SplitOrientation.VERTICAL
            logger.info(f"Creating {orientation.name} split in {target_container_id}")
            
            # Split and get new container ID
            new_id = self.split_manager.split_node(target_container_id, orientation)
            
            if new_id:
                # Determine which container gets the document
                # After split: target_container_id becomes child1, new_id is child2
                if zone in [DropZone.RIGHT, DropZone.BOTTOM]:
                    # Document goes to new (second/child2) container
                    dest_id = new_id
                else:
                    # Document goes to first (child1) container
                    # NOTE: After split, the original target no longer exists as a node!
                    # We need to get child1's ID from the split_manager
                    target_node = self.split_manager.get_node(target_container_id)
                    if target_node and target_node.children:
                        dest_id = target_node.children[0].id  # child1
                    else:
                        logger.error(f"Cannot find child1 after split")
                        self.end_drag()
                        return
                
                # Add document to destination container
                dest_node = self.split_manager.get_node(dest_id)
                if dest_node and dest_node.container_widget:
                    dest_node.container_widget.add_document(
                        document,
                        document.title if hasattr(document, 'title') else "Document"
                    )
                    logger.info(f"✓ Document added to split container: {doc_id} -> {dest_id}")
                else:
                    logger.error(f"Destination container not found: {dest_id}")
            else:
                logger.error(f"Failed to create split")
        
        self.end_drag()
    
    def _move_document(self, doc_id: str, from_container_id: str, to_container_id: str):
        """
        Move document between containers.
        
        Args:
            doc_id: Document ID
            from_container_id: Source container
            to_container_id: Target container
        """
        logger.info(f"Moving document {doc_id}: {from_container_id} -> {to_container_id}")
        
        # Get source and target containers
        source_node = self.split_manager.get_node(from_container_id)
        target_node = self.split_manager.get_node(to_container_id)
        
        if not source_node or not source_node.container_widget:
            logger.error(f"Source container not found: {from_container_id}")
            return
        
        if not target_node or not target_node.container_widget:
            logger.error(f"Target container not found: {to_container_id}")
            return
        
        source_container = source_node.container_widget
        target_container = target_node.container_widget
        
        # Remove from source
        document = source_container.remove_document(doc_id)
        if not document:
            logger.error(f"Document not found in source: {doc_id}")
            return
        
        # Add to target
        target_container.add_document(document, document.title if hasattr(document, 'title') else "Document")
        
        logger.info(f"✓ Document moved successfully: {doc_id}")
    
    def _calculate_preview_rect(self, container_id: str, zone: DropZone) -> QRect:
        """
        Calculate rectangle for drop preview.
        
        Args:
            container_id: Container ID
            zone: Drop zone
            
        Returns:
            Rectangle for preview
        """
        node = self.split_manager.get_node(container_id)
        if not node or not node.container_widget:
            return QRect()
        
        widget = node.container_widget
        w, h = widget.width(), widget.height()
        
        # Calculate preview based on zone
        if zone == DropZone.CENTER:
            # Full container
            return QRect(0, 0, w, h)
        
        elif zone == DropZone.LEFT:
            # Left half
            return QRect(0, 0, w // 2, h)
        
        elif zone == DropZone.RIGHT:
            # Right half
            return QRect(w // 2, 0, w // 2, h)
        
        elif zone == DropZone.TOP:
            # Top half
            return QRect(0, 0, w, h // 2)
        
        elif zone == DropZone.BOTTOM:
            # Bottom half
            return QRect(0, h // 2, w, h // 2)
        
        return QRect()
    
    def _hide_all_previews(self):
        """Hide and delete all preview widgets."""
        for preview in self.previews.values():
            preview.close()
            preview.deleteLater()
        self.previews.clear()
    
    def end_drag(self):
        """Clean up overlays and previews when drag ends."""
        logger.info("Ending drag")
        
        # Close all overlays
        for overlay in self.overlays.values():
            overlay.close()
            overlay.deleteLater()
        self.overlays.clear()
        
        # Close all previews
        self._hide_all_previews()
        
        # Clear drag state
        self.active_drag = None
        
        logger.info("Drag ended, cleanup complete")
