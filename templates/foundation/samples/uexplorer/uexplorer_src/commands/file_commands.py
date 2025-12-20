"""
File Operation Commands

Demonstrates Foundation's CommandBus pattern for decoupled operations.
"""
from typing import List
from loguru import logger


class TagFilesCommand:
    """
    Command to tag multiple files.
    
    Demonstrates:
    - Decoupled operations via CommandBus
    - Testable business logic
    - Potential for undo/redo support
    """
    
    def __init__(self, locator, file_ids: List[str], tag_id: str):
        """
        Initialize tag command.
        
        Args:
            locator: ServiceLocator for accessing TagManager
            file_ids: List of file record IDs to tag
            tag_id: Tag ID to apply
        """
        self.locator = locator
        self.file_ids = file_ids
        self.tag_id = tag_id
    
    async def execute(self) -> dict:
        """
        Execute the tag operation.
        
        Returns:
            dict: Result with success status and count
        """
        from src.ucorefs.tags.manager import TagManager
        from src.core.journal.service import JournalService
        
        logger.info(f"TagFilesCommand: Tagging {len(self.file_ids)} files with tag {self.tag_id}")
        
        tag_manager = self.locator.get_system(TagManager)
        if not tag_manager:
            logger.error("TagManager not available")
            return {'success': False, 'error': 'TagManager not available'}
        
        try:
            # Execute the tagging operation
            await tag_manager.tag_files(self.file_ids, self.tag_id)
            
            logger.info(f"Successfully tagged {len(self.file_ids)} files")
            
            # Audit logging with JournalService
            journal = self.locator.get_system(JournalService)
            if journal:
                await journal.log_event("file_operation", {
                    "action": "tag_files",
                    "file_count": len(self.file_ids),
                    "tag_id": self.tag_id,
                    "file_ids": self.file_ids[:5],  # Log first 5 for audit trail
                    "status": "success"
                })
                logger.debug("Audit log created for tag operation")
            
            return {
                'success': True,
                'file_count': len(self.file_ids),
                'tag_id': self.tag_id
            }
            
        except Exception as e:
            logger.error(f"Failed to tag files: {e}")
            
            # Log failure to journal
            journal = self.locator.get_system(JournalService)
            if journal:
                await journal.log_event("file_operation", {
                    "action": "tag_files",
                    "file_count": len(self.file_ids),
                    "tag_id": self.tag_id,
                    "status": "failed",
                    "error": str(e)
                })
            
            return {
                'success': False,
                'error': str(e)
            }


class UntagFilesCommand:
    """
    Command to remove tags from files.
    
    Demonstrates undo capability.
    """
    
    def __init__(self, locator, file_ids: List[str], tag_id: str):
        self.locator = locator
        self.file_ids = file_ids
        self.tag_id = tag_id
    
    async def execute(self) -> dict:
        """Remove tag from files."""
        from src.ucorefs.tags.manager import TagManager
        from src.core.journal.service import JournalService
        
        logger.info(f"UntagFilesCommand: Removing tag {self.tag_id} from {len(self.file_ids)} files")
        
        tag_manager = self.locator.get_system(TagManager)
        if not tag_manager:
            return {'success': False, 'error': 'TagManager not available'}
        
        try:
            await tag_manager.untag_files(self.file_ids, self.tag_id)
            
            # Audit log
            journal = self.locator.get_system(JournalService)
            if journal:
                await journal.log_event("file_operation", {
                    "action": "untag_files",
                    "file_count": len(self.file_ids),
                    "tag_id": self.tag_id,
                    "status": "success"
                })
            
            return {
                'success': True,
                'file_count': len(self.file_ids),
                'tag_id': self.tag_id
            }
        except Exception as e:
            logger.error(f"Failed to untag files: {e}")
            return {'success': False, 'error': str(e)}


# Example of more complex command with undo support
class DeleteFilesCommand:
    """
    Command to delete files with undo support.
    
    This is a demonstration of how undo/redo could work.
    """
    
    def __init__(self, locator, file_ids: List[str]):
        self.locator = locator
        self.file_ids = file_ids
        self._deleted_records = []
    
    async def execute(self) -> dict:
        """
        Delete files and store for potential undo.
        
        Returns:
            dict: Result with success status
        """
        # Note: This is a demonstration - actual deletion would need
        # to store the full record data for undo
        logger.info(f"DeleteFilesCommand: Would delete {len(self.file_ids)} files")
        logger.warning("Actual deletion not implemented - demonstration only")
        
        return {
            'success': True,
            'action': 'delete',
            'file_count': len(self.file_ids),
            'note': 'Demonstration only - not actually deleting'
        }
    
    async def undo(self) -> dict:
        """
        Undo deletion by restoring records.
        
        Returns:
            dict: Result of undo operation
        """
        logger.info(f"DeleteFilesCommand.undo: Would restore {len(self.file_ids)} files")
        
        return {
            'success': True,
            'action': 'undo_delete',
            'file_count': len(self.file_ids)
        }
