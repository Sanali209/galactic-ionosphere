"""
ContextSyncManager - Central service for reactive property synchronization.

Manages logical 'sync channels' and coordinates property updates between 
different ViewModels without direct coupling.
"""
import weakref
from typing import Dict, List, Any, Optional
from PySide6.QtCore import QObject
from src.core.base_system import BaseSystem
from loguru import logger


class ContextSyncManager(BaseSystem):
    """
    Central service for reactive property synchronization.
    
    This service acts as a mediator for properties marked with a 'sync_channel'.
    When such a property changes in one ViewModel, it is automatically 
    broadcast to all other ViewModels registered on the same channel.
    """
    
    def __init__(self, locator, config):
        """
        Initialize the Sync Manager.
        
        Args:
            locator: ServiceLocator instance.
            config: ConfigManager instance.
        """
        super().__init__(locator, config)
        # registry: { channel_name -> [(weak_vm_ref, property_name), ...] }
        self._registry: Dict[str, List[tuple]] = {}
        
        logger.info("ContextSyncManager initialized")
    
    def register(self, channel: str, vm: QObject, property_name: str) -> None:
        """
        Register a property of a ViewModel to a synchronization channel.
        
        Args:
            channel: Name of the channel to join.
            vm: The ViewModel instance owning the property.
            property_name: Name of the property attribute to sync.
        """
        if channel not in self._registry:
            self._registry[channel] = []
            
        # Check if already registered to avoid duplicates
        for ref, prop in self._registry[channel]:
            if ref() == vm and prop == property_name:
                return
                
        self._registry[channel].append((weakref.ref(vm), property_name))
        
        # --- Advanced Reactivity Setup ---
        # 1. Connect to signals of the INITIAL value (if it's a collection or BindableBase)
        self._connect_value_signals(channel, vm, property_name)
        
        # 2. Listen for when this property itself is replaced with a new instance
        # so we can re-subscribe signals to the new value.
        if hasattr(vm, 'propertyChanged'):
             vm.propertyChanged.connect(
                 lambda prop, val, c=channel, v=vm, p=property_name: 
                 self._on_vm_property_changed(c, v, p, prop, val)
             )
        
        logger.debug(f"Registered {vm.__class__.__name__}.{property_name} to channel: '{channel}'")

    def _on_vm_property_changed(self, channel: str, vm: QObject, target_prop: str, 
                                changed_prop: str, new_val: Any) -> None:
        """Handle when a registered property is replaced with a new instance."""
        if changed_prop == target_prop:
            # The property tied to the sync channel was replaced.
            # Re-subscribe to signals of the NEW value instance.
            self._connect_value_signals(channel, vm, target_prop)

    def _connect_value_signals(self, channel: str, vm: QObject, property_name: str) -> None:
        """Subscribe to internal mutation signals of a property's value."""
        val = getattr(vm, property_name, None)
        if val is None:
            return
            
        # Handle BindableList/BindableDict mutations
        if hasattr(val, 'collectionChanged'):
            # Using partial-like lambda to capture context
            val.collectionChanged.connect(
                lambda col, c=channel, s=vm: self.publish(c, col, s)
            )
            
        # Handle Nested BindableBase property changes (Recursive Observation)
        elif hasattr(val, 'propertyChanged'):
            val.propertyChanged.connect(
                lambda nested_p, nested_v, c=channel, s=vm, p=property_name: 
                self.publish(c, getattr(s, p), s)
            )
        
    def publish(self, channel: str, value: Any, source_vm: QObject) -> None:
        """
        Broadcast a value change to all ViewModels on a channel.
        
        Args:
            channel: The channel name.
            value: The new value to propagate.
            source_vm: The ViewModel that initiated the change.
        """
        if channel not in self._registry:
            return
            
        logger.debug(f"Sync broadcast on '{channel}' from {source_vm.__class__.__name__}")
        
        # Keep track of active references to clean up dead ones
        active_registrations = []
        
        for vm_ref, property_name in self._registry[channel]:
            target_vm = vm_ref()
            
            # Clean up if ViewModel has been garbage collected
            if target_vm is None:
                continue
                
            active_registrations.append((vm_ref, property_name))
            
            # Don't propagate back to the source
            if target_vm is source_vm:
                continue
            
            # Perform synchronized update on target
            try:
                # Check if target already has this instance (common for shared collections)
                old_val = getattr(target_vm, property_name, None)
                
                if old_val is value:
                    # SAME instance - we don't need to call setattr (which would be a no-op 
                    # in BindableProperty), but we MAY need to trigger a local refresh 
                    # signal so the View hears about the mutation.
                    if hasattr(target_vm, 'notify_property_changed'):
                        target_vm.notify_property_changed(property_name, value)
                    continue

                # Set internal update flag to prevent the target from 
                # re-publishing this change back to the manager (infinite loop protection)
                setattr(target_vm, "_mvvm_internal_update", True)
                
                # Set the property value (triggers signals in target_vm)
                setattr(target_vm, property_name, value)
                
            except Exception as e:
                logger.error(f"Failed to sync channel '{channel}' to {target_vm.__class__.__name__}.{property_name}: {e}")
            finally:
                # Always clear the internal update flag
                setattr(target_vm, "_mvvm_internal_update", False)
                
        # Update registry with cleaned list
        self._registry[channel] = active_registrations

    # --- System Interface ---
    
    async def initialize(self):
        """Async initialization (BaseSystem compatibility)."""
        await super().initialize()
        
    async def shutdown(self):
        """Cleanup on application shutdown."""
        self._registry.clear()
        logger.info("ContextSyncManager shut down")
