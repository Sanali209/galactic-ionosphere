"""
Decorator Utilities for Foundation.

Provides syntactic sugar for common patterns.
"""
from typing import Type, TypeVar, Optional, List, Any

T = TypeVar('T')


def system(depends_on: Optional[List[Type]] = None, name: Optional[str] = None):
    """
    Decorator to mark a class as a system.
    
    Args:
        depends_on: List of system types this depends on
        name: System name (defaults to class name)
        
    Usage:
        @system(depends_on=[DatabaseManager])
        class FSService(BaseSystem):
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        cls.depends_on = depends_on or []
        cls._system_name = name or cls.__name__
        return cls
    return decorator


def on_lifecycle(state: str):
    """
    Decorator to mark a function as a lifecycle hook.
    
    Args:
        state: State name ('STARTED', 'STOPPED', etc.)
        
    Usage:
        @on_lifecycle("STARTED")
        def when_started():
            pass
    """
    def decorator(func):
        func._lifecycle_state = state
        return func
    return decorator


def subscribe_event(*event_types: str):
    """
    Decorator to mark a function as an event subscriber.
    
    Args:
        *event_types: Event types to subscribe to
        
    Usage:
        @subscribe_event("file.created", "file.deleted")
        def on_file_event(**data):
            pass
    """
    def decorator(func):
        func._subscribed_events = list(event_types)
        return func
    return decorator
