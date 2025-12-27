"""
Base Plugin class.

All plugins should inherit from Plugin.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..locator import ServiceLocator


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    STARTED = "started"


class Plugin(ABC):
    """
    Base class for all plugins.
    
    Lifecycle: load -> enable -> start -> stop -> disable -> unload
    
    Usage:
        class MyPlugin(Plugin):
            def on_load(self):
                # Initialize resources
                pass
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the plugin.
        
        Args:
            name: Plugin name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.version = "1.0.0"
        self.description = ""
        self.author = ""
        
        self._state = PluginState.UNLOADED
        self._locator: Optional['ServiceLocator'] = None
    
    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state
    
    @property
    def locator(self) -> Optional['ServiceLocator']:
        """Get service locator."""
        return self._locator
    
    def load(self, locator: 'ServiceLocator') -> bool:
        """
        Load the plugin.
        
        Args:
            locator: ServiceLocator instance
            
        Returns:
            True if load succeeded
        """
        if self._state != PluginState.UNLOADED:
            return False
        
        try:
            self._locator = locator
            self.on_load()
            self._state = PluginState.LOADED
            logger.info(f"Plugin loaded: {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin {self.name}: {e}")
            return False
    
    def unload(self) -> bool:
        """Unload the plugin."""
        if self._state != PluginState.LOADED:
            return False
        
        try:
            self.on_unload()
            self._state = PluginState.UNLOADED
            self._locator = None
            logger.info(f"Plugin unloaded: {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload plugin {self.name}: {e}")
            return False
    
    def enable(self) -> bool:
        """Enable the plugin."""
        if self._state != PluginState.LOADED:
            return False
        
        try:
            self.on_enable()
            self._state = PluginState.ENABLED
            return True
        except Exception as e:
            logger.error(f"Failed to enable plugin {self.name}: {e}")
            return False
    
    def disable(self) -> bool:
        """Disable the plugin."""
        if self._state != PluginState.ENABLED:
            return False
        
        try:
            self.on_disable()
            self._state = PluginState.LOADED
            return True
        except Exception as e:
            logger.error(f"Failed to disable plugin {self.name}: {e}")
            return False
    
    def start(self) -> bool:
        """Start the plugin."""
        if self._state != PluginState.ENABLED:
            return False
        
        try:
            self.on_start()
            self._state = PluginState.STARTED
            return True
        except Exception as e:
            logger.error(f"Failed to start plugin {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the plugin."""
        if self._state != PluginState.STARTED:
            return False
        
        try:
            self.on_stop()
            self._state = PluginState.ENABLED
            return True
        except Exception as e:
            logger.error(f"Failed to stop plugin {self.name}: {e}")
            return False
    
    # Override these methods in subclasses
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass
    
    def on_enable(self) -> None:
        """Called when plugin is enabled."""
        pass
    
    def on_disable(self) -> None:
        """Called when plugin is disabled."""
        pass
    
    def on_start(self) -> None:
        """Called when plugin is started."""
        pass
    
    def on_stop(self) -> None:
        """Called when plugin is stopped."""
        pass
    
    def __repr__(self) -> str:
        return f"Plugin({self.name}, state={self._state.value})"
