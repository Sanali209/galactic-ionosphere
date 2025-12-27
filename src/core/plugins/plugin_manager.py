"""
Plugin Manager.

Manages plugin discovery, loading, and lifecycle.
"""
import os
import importlib.util
from typing import Dict, List, Optional, Type, TYPE_CHECKING
from pathlib import Path
from loguru import logger

from .plugin_base import Plugin, PluginState

if TYPE_CHECKING:
    from ..locator import ServiceLocator


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    Usage:
        manager = PluginManager(locator)
        manager.load_from_directory("plugins/")
        manager.enable_all()
        manager.start_all()
    """
    
    def __init__(self, locator: 'ServiceLocator'):
        """
        Initialize the plugin manager.
        
        Args:
            locator: ServiceLocator instance
        """
        self._locator = locator
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
    
    def register_plugin_class(self, name: str, plugin_cls: Type[Plugin]) -> None:
        """
        Register a plugin class.
        
        Args:
            name: Plugin name
            plugin_cls: Plugin class
        """
        self._plugin_classes[name] = plugin_cls
        logger.debug(f"Registered plugin class: {name}")
    
    def load_plugin(self, plugin_path: str) -> Optional[str]:
        """
        Load a plugin from file path.
        
        Args:
            plugin_path: Path to plugin .py file
            
        Returns:
            Plugin name if loaded, None otherwise
        """
        try:
            path = Path(plugin_path)
            if not path.exists() or path.suffix != '.py':
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find Plugin subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Plugin) and 
                    attr is not Plugin):
                    
                    plugin = attr()
                    if plugin.load(self._locator):
                        self._plugins[plugin.name] = plugin
                        return plugin.name
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None
    
    def load_from_directory(self, directory: str) -> List[str]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory containing plugin files
            
        Returns:
            List of loaded plugin names
        """
        loaded = []
        path = Path(directory)
        
        if not path.exists() or not path.is_dir():
            return loaded
        
        for file in path.glob("*.py"):
            if file.stem.startswith("_"):
                continue
            
            name = self.load_plugin(str(file))
            if name:
                loaded.append(name)
        
        logger.info(f"Loaded {len(loaded)} plugins from {directory}")
        return loaded
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> List[Plugin]:
        """Get all loaded plugins."""
        return list(self._plugins.values())
    
    def enable_all(self) -> None:
        """Enable all loaded plugins."""
        for plugin in self._plugins.values():
            if plugin.state == PluginState.LOADED:
                plugin.enable()
    
    def start_all(self) -> None:
        """Start all enabled plugins."""
        for plugin in self._plugins.values():
            if plugin.state == PluginState.ENABLED:
                plugin.start()
    
    def stop_all(self) -> None:
        """Stop all started plugins."""
        for plugin in self._plugins.values():
            if plugin.state == PluginState.STARTED:
                plugin.stop()
    
    def disable_all(self) -> None:
        """Disable all enabled plugins."""
        for plugin in self._plugins.values():
            if plugin.state == PluginState.ENABLED:
                plugin.disable()
    
    def unload_all(self) -> None:
        """Unload all plugins."""
        for plugin in list(self._plugins.values()):
            if plugin.state == PluginState.LOADED:
                if plugin.unload():
                    del self._plugins[plugin.name]
