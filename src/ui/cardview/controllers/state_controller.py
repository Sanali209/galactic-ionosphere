"""
StateController - Manages CardView state persistence.

Saves/restores state per context_id using MongoDB ORM.
"""
from typing import Any, Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.ui.cardview.card_viewmodel import CardViewModel


class StateController:
    """
    Manages CardView state persistence.
    
    State is persisted per context_id, enabling:
    - Different user sorts per folder
    - Preserved algorithm results
    - Collapsed group memory
    
    Example:
        controller = StateController(viewmodel, locator)
        await controller.save_state("folder:123", {...})
        state = await controller.load_state("folder:123")
    """
    
    def __init__(self, viewmodel: 'CardViewModel', locator):
        """
        Initialize state controller.
        
        Args:
            viewmodel: Parent CardViewModel
            locator: ServiceLocator for database access
        """
        self._viewmodel = viewmodel
        self._locator = locator
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    async def save_state(self, context_id: str, state: Dict[str, Any]):
        """
        Save state to database.
        
        Args:
            context_id: Unique context identifier
            state: State dictionary to save
        """
        self._cache[context_id] = state
        
        try:
            # Try to use MongoDB ORM
            from src.core.database.orm import CollectionRecord
            
            record = await CardViewStateRecord.find_one({"context_id": context_id})
            if record:
                for key, value in state.items():
                    setattr(record, key, value)
                await record.save()
            else:
                record = CardViewStateRecord(context_id=context_id, **state)
                await record.save()
            
            logger.debug(f"Saved state for {context_id}")
            
        except ImportError:
            # Fallback to in-memory only
            logger.warning("MongoDB ORM not available, state cached only")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def load_state(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Load state from database.
        
        Args:
            context_id: Context identifier
            
        Returns:
            State dictionary or None
        """
        # Check cache first
        if context_id in self._cache:
            return self._cache[context_id]
        
        try:
            record = await CardViewStateRecord.find_one({"context_id": context_id})
            if record:
                state = {
                    "user_sort_order": record.user_sort_order,
                    "algorithm_hash": record.algorithm_hash,
                    "collapsed_groups": record.collapsed_groups,
                    "thumbnail_size": record.thumbnail_size,
                    "scroll_position": record.scroll_position,
                }
                self._cache[context_id] = state
                logger.debug(f"Loaded state for {context_id}")
                return state
                
        except ImportError:
            logger.warning("MongoDB ORM not available")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        
        return None
    
    async def clear_state(self, context_id: str):
        """Clear persisted state for context."""
        self._cache.pop(context_id, None)
        
        try:
            record = await CardViewStateRecord.find_one({"context_id": context_id})
            if record:
                await record.delete()
            logger.debug(f"Cleared state for {context_id}")
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")


# --- MongoDB ORM Model (optional) ---

try:
    from src.core.database.orm import (
        CollectionRecord, StringField, IntField, ListField
    )
    
    class CardViewStateRecord(CollectionRecord):
        """
        MongoDB ORM model for CardView state.
        """
        _collection_name = "cardview_states"
        
        context_id = StringField(index=True, unique=True)
        user_sort_order = ListField(StringField())
        algorithm_hash = StringField()
        collapsed_groups = ListField(StringField())
        thumbnail_size = IntField(default=200)
        scroll_position = IntField(default=0)

except ImportError:
    # ORM not available, use dummy class
    class CardViewStateRecord:
        """Placeholder when ORM not available."""
        pass
