"""
Layout state serialization for split configurations.
"""
import json
from typing import Dict, Any
from pathlib import Path
from loguru import logger

class LayoutState:
    """
    Handles saving and loading of split layout state.
    """
    @staticmethod
    def save_to_file(split_manager, filepath: str):
        """Save split tree to JSON file."""
        try:
            data = split_manager.to_dict()
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Layout state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save layout state: {e}")
    
    @staticmethod
    def load_from_file(filepath: str):
        """Load split tree from JSON file. Returns SplitManager or None."""
        try:
            if not Path(filepath).exists():
                logger.warning(f"Layout file not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            from .split_manager import SplitManager
            manager = SplitManager.from_dict(data)
            
            logger.info(f"Layout state loaded from {filepath}")
            return manager
        except Exception as e:
            logger.error(f"Failed to load layout state: {e}")
            return None
    
    @staticmethod
    def to_config_dict(split_manager) -> Dict[str, Any]:
        """Convert to dict for embedding in ConfigManager."""
        return split_manager.to_dict()
    
    @staticmethod
    def from_config_dict(data: Dict[str, Any]):
        """Create SplitManager from config dict."""
        from .split_manager import SplitManager
        return SplitManager.from_dict(data)
