"""
Application Lifecycle State Machine.

Manages application state transitions and lifecycle hooks.
"""
from enum import Enum
from typing import Callable, List, Dict
from loguru import logger


class AppState(Enum):
    """Application lifecycle states."""
    CREATED = "created"
    INITIALIZED = "initialized"
    STARTED = "started"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class LifecycleError(Exception):
    """Exception raised for invalid lifecycle transitions."""
    pass


class LifecycleManager:
    """
    Manages application lifecycle state transitions and hooks.
    
    Usage:
        lifecycle = LifecycleManager()
        lifecycle.register_hook(AppState.STARTED, on_started)
        lifecycle.transition_to(AppState.STARTED)
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AppState.CREATED: [AppState.INITIALIZED],
        AppState.INITIALIZED: [AppState.STARTED],
        AppState.STARTED: [AppState.RUNNING, AppState.STOPPING],
        AppState.RUNNING: [AppState.STOPPING],
        AppState.STOPPING: [AppState.STOPPED],
        AppState.STOPPED: [AppState.CREATED],  # Allow restart
    }
    
    def __init__(self):
        """Initialize the lifecycle manager."""
        self._state = AppState.CREATED
        self._hooks: Dict[AppState, List[Callable]] = {s: [] for s in AppState}
        self._listeners: List[Callable[[AppState, AppState], None]] = []
    
    @property
    def state(self) -> AppState:
        """Get current application state."""
        return self._state
    
    def can_transition(self, target: AppState) -> bool:
        """
        Check if can transition to target state.
        
        Args:
            target: Target state
            
        Returns:
            True if transition is valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._state, [])
        return target in valid_targets
    
    def transition_to(self, target: AppState) -> bool:
        """
        Transition to target state.
        
        Args:
            target: Target state
            
        Returns:
            True if transition succeeded
            
        Raises:
            LifecycleError: If transition is invalid
        """
        if not self.can_transition(target):
            raise LifecycleError(
                f"Invalid transition: {self._state.value} -> {target.value}"
            )
        
        old_state = self._state
        self._state = target
        
        logger.info(f"Lifecycle: {old_state.value} -> {target.value}")
        
        # Execute hooks
        self._execute_hooks(target)
        
        # Notify listeners
        self._notify_listeners(old_state, target)
        
        return True
    
    def register_hook(self, state: AppState, hook: Callable) -> None:
        """
        Register a hook to be called when entering a state.
        
        Args:
            state: State to hook
            hook: Callable to execute
        """
        if hook not in self._hooks[state]:
            self._hooks[state].append(hook)
    
    def unregister_hook(self, state: AppState, hook: Callable) -> None:
        """
        Unregister a hook.
        
        Args:
            state: State to unregister from
            hook: Callable to remove
        """
        if hook in self._hooks[state]:
            self._hooks[state].remove(hook)
    
    def add_listener(self, listener: Callable[[AppState, AppState], None]) -> None:
        """
        Add a listener for all state changes.
        
        Args:
            listener: Callable(old_state, new_state)
        """
        if listener not in self._listeners:
            self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable) -> None:
        """Remove a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def _execute_hooks(self, state: AppState) -> None:
        """Execute all hooks for a state."""
        for hook in self._hooks[state]:
            try:
                hook()
            except Exception as e:
                logger.error(f"Hook error for {state.value}: {e}")
    
    def _notify_listeners(self, old: AppState, new: AppState) -> None:
        """Notify all state change listeners."""
        for listener in self._listeners:
            try:
                listener(old, new)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    def reset(self) -> None:
        """Reset to CREATED state (for testing)."""
        self._state = AppState.CREATED
    
    # Convenience properties
    @property
    def is_created(self) -> bool:
        return self._state == AppState.CREATED
    
    @property
    def is_initialized(self) -> bool:
        return self._state == AppState.INITIALIZED
    
    @property
    def is_started(self) -> bool:
        return self._state == AppState.STARTED
    
    @property
    def is_running(self) -> bool:
        return self._state == AppState.RUNNING
    
    @property
    def is_stopping(self) -> bool:
        return self._state == AppState.STOPPING
    
    @property
    def is_stopped(self) -> bool:
        return self._state == AppState.STOPPED
