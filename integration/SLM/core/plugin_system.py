"""
Plugin System
Dynamic plugin loading and management system
"""

import os
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import threading
from SLM.core.singleton import Singleton
from SLM.core.component import Component
from loguru import logger



class Plugin(ABC):
    """
    Base class for all plugins
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the plugin

        Args:
            name: Plugin name (optional)
        """
        self.name = name or self.__class__.__name__
        self._loaded = False
        self._enabled = True
        self._started = False
        self._lock = threading.Lock()
        self.app = None
        self.config = None

    def load(self, app):
        """
        Load the plugin

        Args:
            app: Application instance
        """
        if self._loaded:
            return

        with self._lock:
            try:
                self.app = app
                self.on_load()
                self._loaded = True
                logger.info(f"Plugin {self.name} loaded")
            except Exception as e:
                logger.error(f"Error loading plugin {self.name}: {e}")
                raise

    def unload(self):
        """
        Unload the plugin
        """
        if not self._loaded:
            return

        with self._lock:
            try:
                self.on_unload()
                self._loaded = False
                logger.info(f"Plugin {self.name} unloaded")
            except Exception as e:
                logger.error(f"Error unloading plugin {self.name}: {e}")
                raise

    def enable(self):
        """
        Enable the plugin
        """
        if not self._loaded:
            return

        with self._lock:
            if not self._enabled:
                try:
                    self.on_enable()
                    self._enabled = True
                    logger.info(f"Plugin {self.name} enabled")
                except Exception as e:
                    logger.error(f"Error enabling plugin {self.name}: {e}")

    def disable(self):
        """
        Disable the plugin
        """
        if not self._loaded:
            return

        with self._lock:
            if self._enabled:
                try:
                    self.on_disable()
                    self._enabled = False
                    logger.info(f"Plugin {self.name} disabled")
                except Exception as e:
                    logger.error(f"Error disabling plugin {self.name}: {e}")

    def start(self):
        """
        Start the plugin
        """
        if not self._loaded or not self._enabled:
            return

        with self._lock:
            try:
                self.on_start()
                self._started = True
                logger.info(f"Plugin {self.name} started")
            except Exception as e:
                logger.error(f"Error starting plugin {self.name}: {e}")
                raise

    def stop(self):
        """
        Stop the plugin
        """
        if not self._loaded:
            return

        with self._lock:
            try:
                self.on_stop()
                self._started = False
                logger.info(f"Plugin {self.name} stopped")
            except Exception as e:
                logger.error(f"Error stopping plugin {self.name}: {e}")

    # Abstract methods that subclasses should implement
    def on_load(self):
        """
        Called when plugin is loaded
        """
        pass

    def on_unload(self):
        """
        Called when plugin is unloaded
        """
        pass

    def on_enable(self):
        """
        Called when plugin is enabled
        """
        pass

    def on_disable(self):
        """
        Called when plugin is disabled
        """
        pass

    def on_start(self):
        """
        Called when plugin is started
        """
        pass

    def on_stop(self):
        """
        Called when plugin is stopped
        """
        pass

    def send_message(self, message_type: str, **kwargs):
        """
        Send a message through the message bus

        Args:
            message_type: Type of message
            **kwargs: Message data
        """
        if self.app and hasattr(self.app, 'message_bus'):
            self.app.message_bus.publish(message_type, **kwargs)

    def subscribe_to_message(self, message_type: str, handler):
        """
        Subscribe to a message type

        Args:
            message_type: Message type to subscribe to
            handler: Handler function
        """
        if self.app and hasattr(self.app, 'message_bus'):
            self.app.message_bus.subscribe(message_type, handler)

    def get_component(self, name: str) -> Optional[Component]:
        """
        Get a component from the application

        Args:
            name: Component name

        Returns:
            Component instance or None
        """
        if self.app and hasattr(self.app, 'component_manager'):
            return self.app.component_manager.get_component(name)
        return None

    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded"""
        return self._loaded

    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled"""
        return self._enabled

    @property
    def is_started(self) -> bool:
        """Check if plugin is started"""
        return self._started

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, loaded={self._loaded}, enabled={self._enabled})"

    def __str__(self):
        return f"{self.name} - {'Loaded' if self._loaded else 'Not Loaded'}"


class PluginSystem(Singleton):
    """
    Singleton plugin system - manages dynamic plugin loading and lifecycle
    """

    def __init__(self):
        """
        Initialize the plugin system
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
            
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._plugin_paths: List[str] = []
        self._lock = threading.Lock()
        self.app = None
        self._initialized = True

    def register_plugin_class(self, name: str, plugin_class: Type[Plugin]):
        """
        Register a plugin class

        Args:
            name: Plugin name
            plugin_class: Plugin class
        """
        with self._lock:
            if not issubclass(plugin_class, Plugin):
                raise TypeError("Plugin class must inherit from Plugin")

            self._plugin_classes[name] = plugin_class
            logger.info(f"Registered plugin class: {name}")

    def load_plugin(self, plugin_path: str):
        """
        Load a plugin from file path

        Args:
            plugin_path: Path to plugin file
        """
        plugin_path = os.path.abspath(plugin_path)

        # Check if already loaded
        if plugin_path in self._plugin_paths:
            logger.warning(f"Plugin {plugin_path} already loaded")
            return

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin from {plugin_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, Plugin) and
                    obj != Plugin):
                    plugin_classes.append((name, obj))

            if not plugin_classes:
                logger.warning(f"No plugin classes found in {plugin_path}")
                return

            # Instantiate and register plugins
            for class_name, plugin_class in plugin_classes:
                plugin_name = f"{plugin_path}:{class_name}"
                plugin_instance = plugin_class(class_name)
                self._plugins[plugin_name] = plugin_instance
                self._plugin_paths.append(plugin_path)

                logger.info(f"Loaded plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_path}: {e}")
            raise

    def load_plugins_from_directory(self, directory: str):
        """
        Load all plugins from a directory

        Args:
            directory: Directory containing plugin files
        """
        directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            logger.warning(f"Plugin directory does not exist: {directory}")
            return

        # Find all Python files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                plugin_path = os.path.join(directory, filename)
                try:
                    self.load_plugin(plugin_path)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_path}: {e}")

    def load_plugin_by_name(self, name: str):
        """
        Load a plugin by name

        Args:
            name: Plugin name
        """
        with self._lock:
            if name in self._plugin_classes:
                plugin_class = self._plugin_classes[name]
                plugin_instance = plugin_class()
                plugin_instance.load(self.app)
                self._plugins[name] = plugin_instance
                logger.info(f"Loaded plugin: {name}")
            else:
                raise ValueError(f"Plugin class '{name}' not registered")

    def unload_plugin(self, name: str):
        """
        Unload a plugin

        Args:
            name: Plugin name
        """
        with self._lock:
            if name in self._plugins:
                plugin = self._plugins[name]
                plugin.unload()
                del self._plugins[name]

                # Remove from paths if it's the last plugin from that path
                plugin_path = None
                for path in self._plugin_paths:
                    if name.startswith(path):
                        plugin_path = path
                        break

                if plugin_path:
                    # Check if there are other plugins from the same path
                    remaining_plugins = [p for p in self._plugins.keys() if p.startswith(plugin_path)]
                    if not remaining_plugins:
                        self._plugin_paths.remove(plugin_path)

                logger.info(f"Unloaded plugin: {name}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a plugin by name

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        with self._lock:
            return self._plugins.get(name)

    def get_all_plugins(self) -> Dict[str, Plugin]:
        """
        Get all loaded plugins

        Returns:
            Dictionary of plugin name to plugin instance
        """
        with self._lock:
            return self._plugins.copy()

    def start_plugins(self):
        """
        Start all loaded plugins
        """
        with self._lock:
            for name, plugin in self._plugins.items():
                try:
                    plugin.start()
                except Exception as e:
                    logger.error(f"Error starting plugin {name}: {e}")

    def stop_plugins(self):
        """
        Stop all loaded plugins
        """
        with self._lock:
            for name, plugin in list(self._plugins.items()):
                try:
                    plugin.stop()
                except Exception as e:
                    logger.error(f"Error stopping plugin {name}: {e}")

    def enable_plugin(self, name: str):
        """
        Enable a plugin

        Args:
            name: Plugin name
        """
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enable()

    def disable_plugin(self, name: str):
        """
        Disable a plugin

        Args:
            name: Plugin name
        """
        with self._lock:
            if name in self._plugins:
                self._plugins[name].disable()

    def get_plugin_names(self) -> List[str]:
        """
        Get all plugin names

        Returns:
            List of plugin names
        """
        with self._lock:
            return list(self._plugins.keys())

    def get_plugin_count(self) -> int:
        """
        Get the number of loaded plugins

        Returns:
            Number of plugins
        """
        with self._lock:
            return len(self._plugins)

    def is_plugin_loaded(self, name: str) -> bool:
        """
        Check if a plugin is loaded

        Args:
            name: Plugin name

        Returns:
            True if plugin is loaded
        """
        with self._lock:
            return name in self._plugins

    def get_plugin_status(self) -> Dict[str, Any]:
        """
        Get status information about all plugins

        Returns:
            Dictionary with plugin status information
        """
        with self._lock:
            status = {}
            for name, plugin in self._plugins.items():
                status[name] = {
                    'loaded': plugin.is_loaded,
                    'enabled': plugin.is_enabled,
                    'started': plugin.is_started,
                    'type': type(plugin).__name__
                }
            return status

    def load_plugins(self):
        """
        Load all registered plugins
        """
        with self._lock:
            for name, plugin_class in self._plugin_classes.items():
                try:
                    plugin_instance = plugin_class()
                    self._plugins[name] = plugin_instance
                    logger.info(f"Loaded plugin: {name}")
                except Exception as e:
                    logger.error(f"Error loading plugin {name}: {e}")

    def set_app_reference(self, app):
        """
        Set the application reference for all plugins

        Args:
            app: Application instance
        """
        with self._lock:
            for plugin in self._plugins.values():
                plugin.app = app

    def __repr__(self):
        return f"PluginSystem(plugins={len(self._plugins)})"

    def __str__(self):
        return f"PluginSystem managing {len(self._plugins)} plugins"
