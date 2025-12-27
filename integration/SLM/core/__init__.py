"""
SLM Framework - Pythonic Core
Elegant, singleton-based framework with decorator syntactic sugar
"""

__version__ = "2.0.0"

# Import singleton base
from SLM.core.singleton import Singleton, SingletonMeta

# Import lifecycle
from SLM.core.lifecycle import AppState, LifecycleManager, LifecycleError

# Import core singletons (lazy initialization)
from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.dependency import DependencyManager
from SLM.core.component import Component, ComponentManager
from SLM.core.plugin_system import Plugin, PluginSystem

# Import decorators
from SLM.core.decorators import (
    component, service, inject, auto_inject, subscribe,
    on_app_start, on_app_stop, on_app_initialize, on_app_shutdown,
    on_config_change, cached_property,
    get_registered_components, get_registered_services,
    get_lifecycle_hooks, get_event_subscribers, clear_registries
)

# Module-level singleton instances (lazy creation)
config = Config()
bus = MessageBus()
dependencies = DependencyManager()
components = ComponentManager()
plugins = PluginSystem()


class Core:
    """
    Elegant access point for all core singletons
    Usage: from SLM.core import Core
           Core.config.database.host = "localhost"
           Core.bus.publish("event")
    """
    
    @property
    def config(self) -> Config:
        """Get configuration singleton"""
        return Config()
    
    @property
    def bus(self) -> MessageBus:
        """Get message bus singleton"""
        return MessageBus()
    
    @property
    def dependencies(self) -> DependencyManager:
        """Get dependency manager singleton"""
        return DependencyManager()
    
    @property
    def components(self) -> ComponentManager:
        """Get component manager singleton"""
        return ComponentManager()
    
    @property
    def plugins(self) -> PluginSystem:
        """Get plugin system singleton"""
        return PluginSystem()


# Create Core instance for elegant access
Core = Core()


# Convenience functions
def reset_all():
    """Reset all singletons (useful for testing)"""
    SingletonMeta.reset_all()
    clear_registries()


def get_version() -> str:
    """Get framework version"""
    return __version__


# Export main items
__all__ = [
    # Version
    '__version__',
    'get_version',
    
    # Singleton infrastructure
    'Singleton',
    'SingletonMeta',
    
    # Lifecycle
    'AppState',
    'LifecycleManager',
    'LifecycleError',
    
    # Core singletons (classes)
    'Config',
    'MessageBus',
    'DependencyManager',
    'Component',
    'ComponentManager',
    'Plugin',
    'PluginSystem',
    
    # Module-level instances
    'config',
    'bus',
    'dependencies',
    'components',
    'plugins',
    'Core',
    
    # Decorators
    'component',
    'service',
    'inject',
    'auto_inject',
    'subscribe',
    'on_app_start',
    'on_app_stop',
    'on_app_initialize',
    'on_app_shutdown',
    'on_config_change',
    'cached_property',
    
    # Utilities
    'reset_all',
    'get_registered_components',
    'get_registered_services',
    'get_lifecycle_hooks',
    'get_event_subscribers',
    'clear_registries',
]
