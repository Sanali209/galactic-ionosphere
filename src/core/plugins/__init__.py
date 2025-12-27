"""
Plugin System.

Dynamic plugin loading and management.
"""
from .plugin_base import Plugin, PluginState
from .plugin_manager import PluginManager

__all__ = ['Plugin', 'PluginState', 'PluginManager']
