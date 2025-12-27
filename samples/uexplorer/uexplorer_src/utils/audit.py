"""
Audit Logging Utilities

Helper functions for audit logging using Foundation's JournalService.
"""
from typing import Dict, Any, Optional
from loguru import logger


async def log_user_action(
    locator,
    action: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None
) -> bool:
    """
    Log user action to JournalService.
    
    Args:
        locator: ServiceLocator instance
        action: Action name (e.g., "tag_files", "delete_files")
        details: Additional details to log
        user_id: Optional user identifier
        
    Returns:
        bool: True if logged successfully, False otherwise
        
    Example:
        await log_user_action(
            locator,
            "tag_files",
            {
                "file_count": 5,
                "tag_id": "nature",
                "status": "success"
            }
        )
    """
    from src.core.journal.service import JournalService
    
    try:
        journal = locator.get_system(JournalService)
        if not journal:
            logger.warning("JournalService not available for audit logging")
            return False
        
        # Build event data
        event_data = {
            "action": action,
            **details
        }
        
        if user_id:
            event_data["user_id"] = user_id
        
        # Log to journal
        await journal.log_event("user_action", event_data)
        
        logger.debug(f"Audit log created: {action}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")
        return False


async def log_file_operation(
    locator,
    operation: str,
    file_ids: list,
    success: bool,
    error: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Log file operation to audit trail.
    
    Args:
        locator: ServiceLocator instance
        operation: Operation type (tag, untag, move, delete, etc.)
        file_ids: List of affected file IDs
        success: Whether operation was successful
        error: Error message if failed
        **kwargs: Additional context
        
    Returns:
        bool: True if logged successfully
        
    Example:
        await log_file_operation(
            locator,
            "tag_files",
            file_ids=["id1", "id2"],
            success=True,
            tag_id="landscape"
        )
    """
    details = {
        "file_count": len(file_ids),
        "file_ids": file_ids[:10],  # Log first 10
        "status": "success" if success else "failed",
        **kwargs
    }
    
    if error:
        details["error"] = error
    
    return await log_user_action(locator, operation, details)
