"""
Plugin System.

Dynamic plugin loading and management with Protocol-based interfaces.
"""
from .plugin_base import Plugin, PluginState
from .plugin_manager import PluginManager
from .protocol import ExtractorProtocol, ServiceProtocol, PluginProtocol

__all__ = [
    'Plugin', 
    'PluginState', 
    'PluginManager',
    # Protocols for structural typing
    'ExtractorProtocol',
    'ServiceProtocol', 
    'PluginProtocol',
]

