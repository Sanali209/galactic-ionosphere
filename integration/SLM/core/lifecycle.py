"""
Application Lifecycle State Machine
Manages application state transitions and lifecycle hooks
"""

from enum import Enum, auto
from typing import Callable, List, Dict, Any
import threading
from loguru import logger


class AppState(Enum):
    """
    Application lifecycle states
    """
    CREATED = auto()      # Initial state after instantiation
    CONFIGURED = auto()   # Configuration loaded
    INITIALIZED = auto()  # Dependencies resolved and injected
    STARTED = auto()      # Components started
    RUNNING = auto()      # Main loop running
    STOPPING = auto()     # Shutdown in progress
    STOPPED = auto()      # Components stopped
    SHUTDOWN = auto()     # Cleanup complete


class LifecycleError(Exception):
    """
    Exception raised for invalid lifecycle transitions
    """
    pass


class LifecycleManager:
    """
    Manages application lifecycle state transitions and hooks
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AppState.CREATED: [AppState.CONFIGURED, AppState.INITIALIZED],
        AppState.CONFIGURED: [AppState.INITIALIZED],
        AppState.INITIALIZED: [AppState.STARTED, AppState.SHUTDOWN],
        AppState.STARTED: [AppState.RUNNING, AppState.STOPPING],
        AppState.RUNNING: [AppState.STOPPING],
        AppState.STOPPING: [AppState.STOPPED],
        AppState.STOPPED: [AppState.INITIALIZED, AppState.SHUTDOWN],
        AppState.SHUTDOWN: [AppState.CREATED],  # Allow restart
    }
    
    def __init__(self):
        """
        Initialize the lifecycle manager
        """
        self._state = AppState.CREATED
        self._lock = threading.Lock()
        
        # Lifecycle hooks
        self._hooks: Dict[AppState, List[Callable]] = {
            state: [] for state in AppState
        }
        
        # State change listeners
        self._state_listeners: List[Callable[[AppState, AppState], None]] = []
    
    @property
    def state(self) -> AppState:
        """
        Get current application state
        
        Returns:
            Current state
        """
        return self._state
    
    @property
    def is_created(self) -> bool:
        """Check if in CREATED state"""
        return self._state == AppState.CREATED
    
    @property
    def is_configured(self) -> bool:
        """Check if in CONFIGURED state"""
        return self._state == AppState.CONFIGURED
    
    @property
    def is_initialized(self) -> bool:
        """Check if in INITIALIZED state"""
        return self._state == AppState.INITIALIZED
    
    @property
    def is_started(self) -> bool:
        """Check if in STARTED state"""
        return self._state == AppState.STARTED
    
    @property
    def is_running(self) -> bool:
        """Check if in RUNNING state"""
        return self._state == AppState.RUNNING
    
    @property
    def is_stopping(self) -> bool:
        """Check if in STOPPING state"""
        return self._state == AppState.STOPPING
    
    @property
    def is_stopped(self) -> bool:
        """Check if in STOPPED state"""
        return self._state == AppState.STOPPED
    
    @property
    def is_shutdown(self) -> bool:
        """Check if in SHUTDOWN state"""
        return self._state == AppState.SHUTDOWN
    
    @property
    def can_configure(self) -> bool:
        """Check if can transition to CONFIGURED"""
        return self._can_transition_to(AppState.CONFIGURED)
    
    @property
    def can_initialize(self) -> bool:
        """Check if can transition to INITIALIZED"""
        return self._can_transition_to(AppState.INITIALIZED)
    
    @property
    def can_start(self) -> bool:
        """Check if can transition to STARTED"""
        return self._can_transition_to(AppState.STARTED)
    
    @property
    def can_run(self) -> bool:
        """Check if can transition to RUNNING"""
        return self._can_transition_to(AppState.RUNNING)
    
    @property
    def can_stop(self) -> bool:
        """Check if can transition to STOPPED"""
        return self._can_transition_to(AppState.STOPPING)
    
    def _can_transition_to(self, target_state: AppState) -> bool:
        """
        Check if can transition to target state
        
        Args:
            target_state: Target state
            
        Returns:
            True if transition is valid
        """
        with self._lock:
            valid_targets = self.VALID_TRANSITIONS.get(self._state, [])
            return target_state in valid_targets
    
    def transition_to(self, target_state: AppState) -> None:
        """
        Transition to target state
        
        Args:
            target_state: Target state
            
        Raises:
            LifecycleError: If transition is invalid
        """
        with self._lock:
            if not self._can_transition_to(target_state):
                raise LifecycleError(
                    f"Invalid state transition: {self._state.name} -> {target_state.name}"
                )
            
            old_state = self._state
            self._state = target_state
            
            logger.debug(f"App state transition: {old_state.name} -> {target_state.name}")
            
        # Execute hooks and listeners outside lock
        self._execute_hooks(target_state)
        self._notify_state_change(old_state, target_state)
    
    def register_hook(self, state: AppState, hook: Callable) -> None:
        """
        Register a hook to be called when entering a state
        
        Args:
            state: State to register hook for
            hook: Callable to execute
        """
        with self._lock:
            if hook not in self._hooks[state]:
                self._hooks[state].append(hook)
                logger.debug(f"Registered hook for state {state.name}")
    
    def unregister_hook(self, state: AppState, hook: Callable) -> None:
        """
        Unregister a hook
        
        Args:
            state: State to unregister hook from
            hook: Callable to remove
        """
        with self._lock:
            if hook in self._hooks[state]:
                self._hooks[state].remove(hook)
                logger.debug(f"Unregistered hook for state {state.name}")
    
    def add_state_listener(self, listener: Callable[[AppState, AppState], None]) -> None:
        """
        Add a listener for state changes
        
        Args:
            listener: Callable that receives (old_state, new_state)
        """
        with self._lock:
            if listener not in self._state_listeners:
                self._state_listeners.append(listener)
    
    def remove_state_listener(self, listener: Callable[[AppState, AppState], None]) -> None:
        """
        Remove a state change listener
        
        Args:
            listener: Listener to remove
        """
        with self._lock:
            if listener in self._state_listeners:
                self._state_listeners.remove(listener)
    
    def _execute_hooks(self, state: AppState) -> None:
        """
        Execute all hooks registered for a state
        
        Args:
            state: State that was entered
        """
        hooks = self._hooks.get(state, []).copy()
        for hook in hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Error executing hook for state {state.name}: {e}")
    
    def _notify_state_change(self, old_state: AppState, new_state: AppState) -> None:
        """
        Notify all listeners of state change
        
        Args:
            old_state: Previous state
            new_state: New state
        """
        listeners = self._state_listeners.copy()
        for listener in listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error notifying state listener: {e}")
    
    def reset(self) -> None:
        """
        Reset lifecycle to CREATED state
        """
        with self._lock:
            self._state = AppState.CREATED
            logger.debug("Lifecycle reset to CREATED state")
    
    def get_state_name(self) -> str:
        """
        Get current state name
        
        Returns:
            State name as string
        """
        return self._state.name
    
    def __repr__(self) -> str:
        return f"LifecycleManager(state={self._state.name})"
    
    def __str__(self) -> str:
        return f"Lifecycle: {self._state.name}"
