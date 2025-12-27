"""
Component System
Base classes for system components and component management
"""

from typing import Dict, Any, Optional, List, Type, TypeVar, cast
from abc import ABC
import threading
import time
from loguru import logger

from SLM.core.singleton import Singleton
from SLM.core.message_bus import MessageBus

T = TypeVar('T', bound='Component')


class Component(ABC):
    """
    Base class for all system components
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the component

        Args:
            name: Component name (optional)
        """
        self.name = name or self.__class__.__name__
        self._initialized = False
        self._started = False
        self._running = False
        self._lock = threading.Lock()
        self.message_bus: Optional[MessageBus] = None
        self.config = None

    def initialize(self):
        """
        Initialize the component
        """
        if self._initialized:
            return

        with self._lock:
            try:
                self.on_initialize()
                self._initialized = True
                logger.debug(f"Component {self.name} initialized")
            except Exception as e:
                logger.error(f"Error initializing component {self.name}: {e}")
                raise

    def start(self):
        """
        Start the component
        """
        if not self._initialized:
            self.initialize()

        if self._started:
            return

        with self._lock:
            try:
                self.on_start()
                self._started = True
                logger.info(f"Component {self.name} started")
            except Exception as e:
                logger.error(f"Error starting component {self.name}: {e}")
                raise

    def stop(self):
        """
        Stop the component
        """
        if not self._started:
            return

        with self._lock:
            try:
                self.on_stop()
                self._started = False
                logger.debug(f"Component {self.name} stopped")
            except Exception as e:
                logger.error(f"Error stopping component {self.name}: {e}")
                raise

    def update(self):
        """
        Update the component (called regularly)
        """
        if not self._started:
            return

        try:
            self.on_update()
        except Exception as e:
            logger.error(f"Error updating component {self.name}: {e}")

    def shutdown(self):
        """
        Shutdown the component
        """
        if self._running:
            self.stop()

        with self._lock:
            try:
                self.on_shutdown()
                self._initialized = False
                logger.debug(f"Component {self.name} shutdown")
            except Exception as e:
                logger.error(f"Error shutting down component {self.name}: {e}")

    def on_config_changed(self, key: str, old_value, new_value):
        """
        Called when a configuration change occurs that might affect this component.
        Components should override this method to handle config changes.

        Args:
            key: The configuration key that changed
            old_value: Previous value
            new_value: New value
        """
        # Default implementation does nothing - components should override
        pass

    def request_reinitialization(self, reason: str = "Configuration change requires reinitialization"):
        """
        Request full application reinitialization because this component
        cannot handle the config change dynamically.

        Args:
            reason: Reason for requiring reinitialization
        """
        if self.message_bus:
            self.message_bus.publish(
                'reinit.required',
                component=self.name,
                component_type=self.__class__.__name__,
                reason=reason,
                timestamp=time.time()
            )
            logger.debug(f"Component {self.name} requested reinitialization: {reason}")

    def setup_config_watching(self):
        """
        Setup subscription to config change events.
        Components can call this explicitly to watch for config changes.
        """
        if self.message_bus and not hasattr(self, '_config_watching_setup'):
            # Mark that we've set up watching to avoid duplicate subscriptions
            self._config_watching_setup = True
            # Subscribe to config change events
            self.subscribe_to_message('config.changed', self._handle_config_change)
            logger.debug(f"{self.name} set up config watching")

    def _handle_config_change(self, event_type: str, **data):
        """
        Internal handler for config change events.
        Routes to the user-overridable method.
        """
        key = data.get('key')
        if key is not None:
            self.on_config_changed(key, data.get('old_value'), data.get('new_value'))

    # Abstract methods that subclasses should implement
    def on_initialize(self):
        """
        Called when component is initialized
        """
        pass

    def on_start(self):
        """
        Called when component is started
        """
        pass

    def on_stop(self):
        """
        Called when component is stopped
        """
        pass

    def on_update(self):
        """
        Called when component is updated
        """
        pass

    def on_shutdown(self):
        """
        Called when component is shutdown
        """
        pass

    def send_message(self, message_type: str, **kwargs):
        """
        Send a message through the message bus

        Args:
            message_type: Type of message
            **kwargs: Message data
        """
        if self.message_bus:
            self.message_bus.publish(message_type, **kwargs)

    def subscribe_to_message(self, message_type: str, handler):
        """
        Subscribe to a message type

        Args:
            message_type: Message type to subscribe to
            handler: Handler function
        """
        if self.message_bus:
            self.message_bus.subscribe(message_type, handler)

    def unsubscribe_from_message(self, message_type: str, handler):
        """
        Unsubscribe from a message type

        Args:
            message_type: Message type to unsubscribe from
            handler: Handler function
        """
        if self.message_bus:
            self.message_bus.unsubscribe(message_type, handler)

    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized

    @property
    def is_started(self) -> bool:
        """Check if component is started"""
        return self._started

    @property
    def is_running(self) -> bool:
        """Check if component is running"""
        return self._running

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, initialized={self._initialized}, started={self._started})"

    def __str__(self):
        return f"{self.name} - {'Running' if self._started else 'Stopped'}"


class ComponentManager(Singleton):
    """
    Singleton component manager - manages and coordinates system components
    """

    def __init__(self):
        """
        Initialize the component manager
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
            
        self._components: Dict[str, Component] = {}
        self._component_types: Dict[Type[Component], List[str]] = {}
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._updating = False
        self._initialized = True

    def register_component(self, name: str, component: Component):
        """
        Register a component

        Args:
            name: Component name
            component: Component instance
        """
        with self._lock:
            if name in self._components:
                raise ValueError(f"Component with name '{name}' already exists")

            self._components[name] = component
            component_type = type(component)
            if component_type not in self._component_types:
                self._component_types[component_type] = []
            self._component_types[component_type].append(name)

            logger.debug(f"Registered component: {name} ({component_type.__name__})")

    def unregister_component(self, name: str):
        """
        Unregister a component

        Args:
            name: Component name
        """
        with self._lock:
            if name in self._components:
                component = self._components[name]
                component.shutdown()
                del self._components[name]

                # Remove from type registry
                component_type = type(component)
                if component_type in self._component_types:
                    if name in self._component_types[component_type]:
                        self._component_types[component_type].remove(name)

                logger.debug(f"Unregistered component: {name}")

    def get_component(self, name: str) -> Optional[Component]:
        """
        Get a component by name

        Args:
            name: Component name

        Returns:
            Component instance or None
        """
        with self._lock:
            return self._components.get(name)

    def get_components_by_type(self, component_type: Type[T]) -> List[T]:
        """
        Get all components of a specific type

        Args:
            component_type: Component type

        Returns:
            List of components of the specified type
        """
        with self._lock:
            component_names = self._component_types.get(component_type, [])
            components = [self._components[name] for name in component_names if name in self._components]
            return cast(List[T], components)

    def get_component_names(self) -> List[str]:
        """
        Get all component names

        Returns:
            List of component names
        """
        with self._lock:
            return list(self._components.keys())

    def get_component_types(self) -> List[Type[Component]]:
        """
        Get all registered component types

        Returns:
            List of component types
        """
        with self._lock:
            return list(self._component_types.keys())

    def initialize_all(self):
        """
        Initialize all registered components
        """
        with self._lock:
            for name, component in self._components.items():
                try:
                    component.initialize()
                except Exception as e:
                    logger.error(f"Error initializing component {name}: {e}")

    def start_all(self):
        """
        Start all registered components
        """
        with self._lock:
            for name, component in self._components.items():
                try:
                    component.start()
                except Exception as e:
                    logger.error(f"Error starting component {name}: {e}")

    def stop_all(self):
        """
        Stop all registered components
        """
        with self._lock:
            for name, component in self._components.items():
                try:
                    component.stop()
                except Exception as e:
                    logger.error(f"Error stopping component {name}: {e}")

    def shutdown_all(self):
        """
        Shutdown all registered components
        """
        with self._lock:
            for name, component in list(self._components.items()):
                try:
                    component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component {name}: {e}")

    def update_all(self):
        """
        Update all registered components
        """
        with self._lock:
            for name, component in self._components.items():
                try:
                    component.update()
                except Exception as e:
                    logger.error(f"Error updating component {name}: {e}")

    def start_updating(self, interval: float = 0.1):
        """
        Start automatic updating of all components

        Args:
            interval: Update interval in seconds
        """
        if self._updating:
            return

        self._updating = True

        def update_loop():
            while self._updating:
                self.update_all()
                time.sleep(interval)

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        logger.debug("Started component updating")

    def stop_updating(self):
        """
        Stop automatic updating of components
        """
        self._updating = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        logger.debug("Stopped component updating")

    def get_component_count(self) -> int:
        """
        Get the total number of registered components

        Returns:
            Number of components
        """
        with self._lock:
            return len(self._components)

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about all components

        Returns:
            Dictionary with component status information
        """
        with self._lock:
            status = {}
            for name, component in self._components.items():
                status[name] = {
                    'type': type(component).__name__,
                    'initialized': component.is_initialized,
                    'started': component.is_started,
                    'running': component.is_running
                }
            return status

    def __repr__(self):
        return f"ComponentManager(components={len(self._components)})"

    def __str__(self):
        return f"ComponentManager managing {len(self._components)} components"
