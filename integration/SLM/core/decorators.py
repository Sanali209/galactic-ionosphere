"""
Decorator Utilities for SLM Framework
Provides syntactic sugar for common patterns
"""

from typing import Callable, Type, Optional, Any, TypeVar, get_type_hints
from functools import wraps
import inspect
from loguru import logger

T = TypeVar('T')

# Global registry for decorated items
_component_registry: dict = {}
_service_registry: dict = {}
_lifecycle_hooks: dict = {
    'on_start': [],
    'on_stop': [],
    'on_initialize': [],
    'on_shutdown': []
}
_event_subscribers: dict = {}


def component(name: Optional[str] = None, auto_register: bool = True):
    """
    Decorator to mark a class as a component
    
    Args:
        name: Component name (defaults to class name)
        auto_register: Whether to auto-register with ComponentManager
        
    Usage:
        @component(name="MyComponent")
        class MyComponent:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        component_name = name or cls.__name__
        cls._component_name = component_name
        cls._auto_register = auto_register
        
        if auto_register:
            _component_registry[component_name] = cls
            logger.debug(f"Registered component: {component_name}")
        
        return cls
    return decorator


def service(singleton: bool = True, name: Optional[str] = None):
    """
    Decorator to mark a class as a service
    
    Args:
        singleton: Whether service should be singleton
        name: Service name (defaults to class name)
        
    Usage:
        @service(singleton=True)
        class MyService:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        service_name = name or cls.__name__
        cls._is_service = True
        cls._service_singleton = singleton
        cls._service_name = service_name
        
        _service_registry[service_name] = {
            'class': cls,
            'singleton': singleton
        }
        
        logger.debug(f"Registered service: {service_name} (singleton={singleton})")
        
        return cls
    return decorator


def inject(**dependencies):
    """
    Decorator to inject dependencies into a function or method
    
    Args:
        **dependencies: Mapping of parameter name to type
        
    Usage:
        @inject(config=Config, bus=MessageBus)
        def my_function(config, bus):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import here to avoid circular dependency
            from SLM.core import dependencies as dep_manager
            
            # Inject dependencies that aren't already provided
            for param_name, param_type in dependencies.items():
                if param_name not in kwargs:
                    try:
                        if dep_manager.has_service(param_type):
                            kwargs[param_name] = dep_manager.get_service(param_type)
                    except Exception as e:
                        logger.warning(f"Failed to inject {param_name}: {e}")
            
            return func(*args, **kwargs)
        
        wrapper._injected_deps = dependencies
        return wrapper
    return decorator


def auto_inject(func: Callable) -> Callable:
    """
    Automatically inject dependencies based on type hints
    
    Usage:
        @auto_inject
        def my_function(config: Config, bus: MessageBus):
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Import here to avoid circular dependency
        from SLM.core import dependencies as dep_manager
        
        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Inject dependencies for parameters with type hints
        for param_name, param in sig.parameters.items():
            if param_name in hints and param_name not in kwargs:
                param_type = hints[param_name]
                try:
                    if dep_manager.has_service(param_type):
                        kwargs[param_name] = dep_manager.get_service(param_type)
                except Exception as e:
                    logger.debug(f"Could not auto-inject {param_name}: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper


def subscribe(*events):
    """
    Decorator to subscribe a function to message bus events
    
    Args:
        *events: Event types to subscribe to
        
    Usage:
        @subscribe("config.changed", "app.started")
        def on_event(event_type, **data):
            pass
    """
    def decorator(func: Callable) -> Callable:
        func._subscribed_events = events
        
        # Register in global registry
        for event in events:
            if event not in _event_subscribers:
                _event_subscribers[event] = []
            _event_subscribers[event].append(func)
        
        logger.debug(f"Registered event subscriber: {func.__name__} for {events}")
        
        # The actual subscription happens when the message bus is available
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._subscribed_events = events
        return wrapper
    return decorator


def on_app_start(func: Callable) -> Callable:
    """
    Decorator to mark a function to be called on app start
    
    Usage:
        @on_app_start
        def initialize_database():
            pass
    """
    _lifecycle_hooks['on_start'].append(func)
    logger.debug(f"Registered on_start hook: {func.__name__}")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def on_app_stop(func: Callable) -> Callable:
    """
    Decorator to mark a function to be called on app stop
    
    Usage:
        @on_app_stop
        def cleanup_database():
            pass
    """
    _lifecycle_hooks['on_stop'].append(func)
    logger.debug(f"Registered on_stop hook: {func.__name__}")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def on_app_initialize(func: Callable) -> Callable:
    """
    Decorator to mark a function to be called on app initialization
    
    Usage:
        @on_app_initialize
        def setup_logging():
            pass
    """
    _lifecycle_hooks['on_initialize'].append(func)
    logger.debug(f"Registered on_initialize hook: {func.__name__}")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def on_app_shutdown(func: Callable) -> Callable:
    """
    Decorator to mark a function to be called on app shutdown
    
    Usage:
        @on_app_shutdown
        def final_cleanup():
            pass
    """
    _lifecycle_hooks['on_shutdown'].append(func)
    logger.debug(f"Registered on_shutdown hook: {func.__name__}")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def on_config_change(key: Optional[str] = None):
    """
    Decorator to subscribe to configuration changes
    
    Args:
        key: Specific config key to watch (None for all changes)
        
    Usage:
        @on_config_change("database.host")
        def on_db_host_changed(old_value, new_value):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(event_type: str, **data):
            config_key = data.get('key')
            
            # If specific key is set, only call if it matches
            if key is None or config_key == key:
                old_value = data.get('old_value')
                new_value = data.get('new_value')
                return func(old_value, new_value)
        
        # Subscribe to config.changed event
        wrapper._subscribed_events = ('config.changed',)
        wrapper._config_key_filter = key
        
        if 'config.changed' not in _event_subscribers:
            _event_subscribers['config.changed'] = []
        _event_subscribers['config.changed'].append(wrapper)
        
        logger.debug(f"Registered config change handler: {func.__name__} for key={key}")
        
        return wrapper
    return decorator


def cached_property(func: Callable) -> property:
    """
    Decorator for a cached property (computed once, then cached)
    
    Usage:
        @cached_property
        def expensive_computation(self):
            return compute()
    """
    attr_name = f'_cached_{func.__name__}'
    
    @wraps(func)
    def getter(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return property(getter)


def get_registered_components():
    """Get all registered components"""
    return _component_registry.copy()


def get_registered_services():
    """Get all registered services"""
    return _service_registry.copy()


def get_lifecycle_hooks(hook_type: str):
    """Get lifecycle hooks of a specific type"""
    return _lifecycle_hooks.get(hook_type, []).copy()


def get_event_subscribers(event_type: Optional[str] = None):
    """Get event subscribers"""
    if event_type:
        return _event_subscribers.get(event_type, []).copy()
    return _event_subscribers.copy()


def clear_registries():
    """Clear all registries (useful for testing)"""
    _component_registry.clear()
    _service_registry.clear()
    for key in _lifecycle_hooks:
        _lifecycle_hooks[key].clear()
    _event_subscribers.clear()
    logger.debug("Cleared all decorator registries")
