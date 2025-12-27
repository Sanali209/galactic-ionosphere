"""
Enhanced Configuration Class with Dynamic Change Reactivity
Allows flexible access and modification of configuration values through:
- Attribute-style access (config.section.key)
- Dot-separated strings (config.get('section.subsection.key'))
- Direct key access (config['key'])
- Dynamic change notifications via message bus
"""

import functools
import time
from typing import Any, Optional, TYPE_CHECKING

from SLM.core.singleton import Singleton

if TYPE_CHECKING:
    from SLM.core.message_bus import MessageBus


class Config(Singleton):
    """
    Singleton configuration manager with attribute-style access and dynamic change reactivity.
    """

    def __init__(self, initial_data=None):
        """
        Initializes the Config object.

        Args:
            initial_data (dict, optional): Initial configuration data. Defaults to None.
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        
        if initial_data is None:
            initial_data = {}
        self._data = self._deep_transform(initial_data)
        self.message_bus: Optional['MessageBus'] = None  # Injected by dependency system
        self._change_tracking_enabled = True
        self._initialized = True

    def _deep_transform(self, data):
        """
        Recursively transforms dictionaries into Config objects.
        """
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    result[k] = Config({k2: self._deep_transform(v2) for k2, v2 in v.items()})
                elif isinstance(v, list):
                    result[k] = [self._deep_transform(i) for i in v]
                else:
                    result[k] = v
            return result
        elif isinstance(data, list):
            return [self._deep_transform(i) for i in data]
        else:
            return data

    def __getattr__(self, name):
        """
        Allows accessing configuration values as attributes (e.g., config.section.key).
        """
        if name in self._data:
            return self._data[name]
        # Return a new Config object for a non-existent attribute to allow chaining
        # for setting nested values.
        new_config = Config()
        self._data[name] = new_config
        return new_config

    def __setattr__(self, name, value):
        """
        Allows setting configuration values as attributes with change tracking.
        """
        if name == '_data':
            super().__setattr__(name, value)
        elif name.startswith('_') or name in ['message_bus', '_change_tracking_enabled']:
            # Internal or special attributes - set directly
            super().__setattr__(name, value)
        else:
            # Configuration value being set - track change and publish event
            old_value = getattr(self, name, None)
            super().__setattr__(name, self._deep_transform(value))
            # Publish change for configuration changes
            if self._change_tracking_enabled and self.message_bus:
                self._publish_config_change(name, old_value, value)

    def get(self, key, default=None):
        """
        Retrieves a configuration value using a dot-separated key.

        Args:
            key (str): The dot-separated key (e.g., 'section.subsection.key').
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        try:
            return functools.reduce(lambda d, k: d[k], key.split('.'), self._data)
        except (KeyError, TypeError):
            return default

    def __repr__(self):
        return repr(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = self._deep_transform(value)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        """Return the keys of the configuration."""
        if isinstance(self._data, dict):
            return self._data.keys()
        return []

    def values(self):
        """Return the values of the configuration."""
        if isinstance(self._data, dict):
            return self._data.values()
        return []

    def items(self):
        """Return the items of the configuration."""
        if isinstance(self._data, dict):
            return self._data.items()
        return []

    def to_dict(self):
        """Convert the configuration to a regular dictionary."""
        if isinstance(self._data, dict):
            return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self._data.items()}
        return self._data

    def update(self, other):
        """Update this configuration with values from another Config or dict."""
        if isinstance(other, Config):
            other = other.to_dict()
        elif isinstance(other, dict):
            pass
        else:
            raise TypeError("Can only update with Config or dict")

        def _update_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], Config):
                    _update_recursive(target[key]._data, value)
                else:
                    target[key] = self._deep_transform(value)

        _update_recursive(self._data, other)

    def merge(self, other):
        """Merge another configuration into this one."""
        self.update(other)

    def clear(self):
        """Clear all configuration values."""
        self._data.clear()

    def copy(self):
        """Create a copy of this configuration."""
        return Config(self.to_dict())

    def _publish_config_change(self, key: str, old_value: Any, new_value: Any):
        """
        Publish a configuration change event via message bus.

        Args:
            key: The configuration key that changed
            old_value: Previous value
            new_value: New value
        """
        if not self._change_tracking_enabled or not self.message_bus:
            return

        self.message_bus.publish(
            'config.changed',
            key=key,
            old_value=old_value,
            new_value=new_value,
            timestamp=time.time()
        )

    def set_value(self, key: str, value: Any, publish_change: bool = True):
        """
        Set a configuration value and optionally publish change event.

        Args:
            key: Dot-separated configuration key
            value: New value to set
            publish_change: Whether to publish change event
        """
        old_value = self.get(key)
        new_value = value

        def _set_nested(target, keys, value):
            current_key = keys[0]
            if len(keys) == 1:
                # Set the final value
                target[current_key] = self._deep_transform(value)
                if publish_change:
                    self._publish_config_change('.'.join(keys), old_value, new_value)
            else:
                # Navigate deeper
                if current_key not in target:
                    target[current_key] = Config()
                elif not isinstance(target[current_key], Config):
                    target[current_key] = Config()
                _set_nested(target[current_key]._data, keys[1:], value)

        _set_nested(self._data, key.split('.'), value)

    def disable_change_tracking(self):
        """Temporarily disable change event publishing."""
        self._change_tracking_enabled = False

    def enable_change_tracking(self):
        """Re-enable change event publishing."""
        self._change_tracking_enabled = True

    def set_with_tracking_disabled(self, other):
        """Update configuration without publishing change events."""
        was_enabled = self._change_tracking_enabled
        self._change_tracking_enabled = False
        try:
            self.update(other)
        finally:
            self._change_tracking_enabled = was_enabled

    def handle_reinitialization_request(self):
        """
        Handle a reinitialization request from a component.
        This method should be called by the App when it receives a 'reinit.required' event.
        """
        if not self.message_bus:
            return

        # Publish reinitialization event to notify all components
        self.message_bus.publish('app.reinitializing')

        # The app should handle the actual reinitialization process
