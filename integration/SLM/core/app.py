"""
Main Application Class
Orchestrates all system components and manages application lifecycle using dependency injection
"""

import threading
import time
from typing import Dict, Any, Optional, TypeVar

import logging
logger = logging.getLogger(__name__)

from SLM.core.component import ComponentManager
from SLM.core.config import Config
from SLM.core.dependency import DependencyManager
from SLM.core.message_bus import MessageBus
from SLM.core.plugin_system import PluginSystem

T = TypeVar('T')


class App:
    """
    Main application orchestrator that manages all system components using dependency injection 11
    """

    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the application with configuration and dependency injection

        Args:
            config_data: Initial configuration data
        """
        # Get or create service locator
        self._dependency_manager = DependencyManager()
        self._dependency_manager.register_singleton(DependencyManager, self._dependency_manager)
        # Application state
        self._running = False
        self._initialized = False
        self._reinitialization_in_progress = False
        self._lock = threading.Lock()

        # Register self as application service
        self._dependency_manager.register_singleton(App, self)

        # Setup core services
        self._setup_core_services(config_data or {})

    def _setup_core_services(self, config_data: Dict[str, Any]) -> None:
        """
        Setup and register core services
        
        Args:
            config_data: Initial configuration data
        """
        # Create and register configuration service
        config_service = Config(config_data)
        self._dependency_manager.register_singleton(Config, config_service)

        # Create and register message bus service
        message_bus_service = MessageBus()
        self._dependency_manager.register_singleton(MessageBus, message_bus_service)

        # Create and register component manager service
        component_manager_service = ComponentManager()
        self._dependency_manager.register_singleton(ComponentManager, component_manager_service)

        # Create and register plugin system service
        plugin_system_service = PluginSystem()
        self._dependency_manager.register_singleton(PluginSystem, plugin_system_service)

    def run(self):
        """
        Run the application synchronously
        """
        try:
            self.initialize()
            self.start()

            # Main application loop
            logger.info("Application running... Press Ctrl+C to stop")
            try:
                # Simple loop - wait for interrupt
                while self._running:
                    time.sleep(1.0)  # Simple ping every second
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self._running = False

        except KeyboardInterrupt:
            logger.info("Received interrupt signal during initialization")
            self._running = False
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            if self._running:
                self._running = False
            self.stop()

    def initialize(self):
        """
        Initialize the application
        """
        if self._initialized:
            return

        logger.info("Initializing application...")

        try:
            # Disable config change tracking during initialization to prevent loops
            config_service = self._dependency_manager.get_service(Config)
            config_service.disable_change_tracking()

            # Get services from locator
            plugin_system = self._dependency_manager.get_service(PluginSystem)
            message_bus = self._dependency_manager.get_service(MessageBus)

            # Setup dependencies (injects message_bus, etc. into components)
            # This also initializes components AFTER dependencies are injected
            self._dependency_manager.setup_dependencies()

            # Load plugins (but don't start them yet)
            plugin_system.load_plugins()

            # Subscribe to reinitialization requests
            self._setup_reinit_listener()

            # Re-enable config change tracking after initialization
            config_service.enable_change_tracking()

            self._initialized = True
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            raise

    def _setup_reinit_listener(self):
        """
        Setup listener for component reinitialization requests
        """
        message_bus = self._dependency_manager.get_service(MessageBus)
        message_bus.subscribe('reinit.required', self._handle_reinit_request)

    def _handle_reinit_request(self, event_type: str, **data):
        """
        Handle a reinitialization request from a component

        Args:
            event_type: Event type (should be 'reinit.required')
            **data: Event data containing component info and reason
        """
        if self._reinitialization_in_progress:
            logger.debug(f"Ignoring reinitialization request - already processing: {data.get('component', 'unknown')}")
            return

        logger.info(f"Reinitialization requested by {data.get('component', 'unknown')}: {data.get('reason', 'no reason')}")

        # Handle the reinitialization request
        self.handle_reinitialization_request(data)

    def handle_reinitialization_request(self, request_data: Dict[str, Any]):
        """
        Handle full application reinitialization

        Args:
            request_data: Information about the reinitialization request
        """
        component_name = request_data.get('component', 'unknown')
        reason = request_data.get('reason', 'no reason given')

        logger.info(f"Performing full application reinitialization due to {component_name}: {reason}")

        # Prevent recursive reinitialization requests during the process
        self._reinitialization_in_progress = True

        try:
            # Disable config change tracking during reinitialization to prevent loops
            config_service = self._dependency_manager.get_service(Config)
            config_service.disable_change_tracking()

            # Stop the application
            self.stop()

            # Mark as not initialized to force full reinit
            self._initialized = False

            # Reinitialize the application
            self.initialize()

            # Re-enable config change tracking
            config_service.enable_change_tracking()

            # Restart the application
            self.start()

            logger.info(f"Application reinitialized successfully due to {component_name}")

        except Exception as e:
            logger.error(f"Error during application reinitialization: {e}")
            # Attempt to restart with original state
            try:
                self.start()
            except Exception as restart_error:
                logger.critical(f"Critical error: Could not restart application: {restart_error}")
                raise
        finally:
            self._reinitialization_in_progress = False

    def start(self):
        """
        Start the application
        """
        if not self._initialized:
            logger.debug("Start requested - app not initialized, initializing first")
            self.initialize()

        if self._running:
            logger.debug("Application is already running, skipping start")
            return

        start_time = time.time()
        logger.info("Starting application...")

        try:
            # Start all components and services (after dependencies are injected)
            self._dependency_manager.start_all()

            # Start plugins
            plugin_system = self._dependency_manager.get_service(PluginSystem)
            plugin_system.start_plugins()

            self._running = True

            duration = time.time() - start_time
            logger.info(f"Application started successfully in {duration:.3f}s")

        except Exception as e:
            logger.error(f"Error starting application: {e}")
            raise

    def stop(self):
        """
        Stop the application
        """
        if not self._running:
            return

        logger.info("Stopping application...")

        try:
            # Get services from locator
            plugin_system = self._dependency_manager.get_service(PluginSystem)
            component_manager = self._dependency_manager.get_service(ComponentManager)

            # Stop plugins first
            plugin_system.stop_plugins()

            # Stop components
            component_manager.stop_all()
            self._dependency_manager.stop_all()

            self._running = False
            logger.info("Application stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
            raise

    def configure(self, config_data: Dict[str, Any]):
        """
        Update configuration

        Args:
            config_data: New configuration data
        """
        config_service = self._dependency_manager.get_service(Config)
        config_service.update(config_data)

    def register_component(self, component: Any):
        """
        Register a new component

        Args:
            component: Component instance
        """
        # Register with component's type, not string name
        self._dependency_manager.register_service(type(component), component)

    def load_plugin(self, plugin_path: str):
        """
        Load a plugin

        Args:
            plugin_path: Path to plugin file or directory
        """
        plugin_system = self._dependency_manager.get_service(PluginSystem)
        plugin_system.load_plugin(plugin_path)

    @property
    def is_running(self) -> bool:
        """Check if application is running"""
        return self._running

    @property
    def is_initialized(self) -> bool:
        """Check if application is initialized"""
        return self._initialized
