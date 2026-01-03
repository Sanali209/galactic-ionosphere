"""
Bootstrap helpers for Foundation applications.

Simplifies application setup and initialization.
"""
import sys
import asyncio
from abc import ABC, abstractmethod
from typing import Type, Optional, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .bootstrap import ApplicationBuilder

from PySide6.QtWidgets import QApplication
from qasync import QEventLoop
from loguru import logger

from .locator import ServiceLocator, sl
from .base_system import BaseSystem
from .database.manager import DatabaseManager
from .commands.bus import CommandBus
from .journal.service import JournalService
from .assets.manager import AssetManager
from .tasks.system import TaskSystem


class SystemBundle(ABC):
    """
    Base class for grouping related systems into reusable bundles.
    
    Bundles encapsulate the registration of multiple systems in their
    required dependency order, reducing verbosity in entry points.
    
    Example:
        class MyBundle(SystemBundle):
            def register(self, builder: "ApplicationBuilder") -> None:
                builder.add_system(ServiceA)
                builder.add_system(ServiceB)
    """
    
    @abstractmethod
    def register(self, builder: "ApplicationBuilder") -> None:
        """
        Register all systems in this bundle.
        
        Args:
            builder: The ApplicationBuilder to register systems with
        """
        pass


class ApplicationBuilder:
    """
    Fluent builder for Foundation applications.
    
    Example:
        app = (ApplicationBuilder("My App", "config.json")
               .with_default_systems()
               .add_system(MyCustomService)
               .build())
    """
    
    def __init__(self, name: str = "Foundation App", config_path: str = "config.json"):
        """
        Initialize application builder.
        
        Args:
            name: Application name
            config_path: Path to config.json file
        """
        self.name = name
        self.config_path = config_path
        self._systems: List[Type[BaseSystem]] = []
        self._use_default_systems = True
        self._logging_configured = False
        
    def with_default_systems(self, enable: bool = True):
        """
        Include default foundation systems.
        
        Default systems:
        - DatabaseManager
        - CommandBus
        - JournalService
        - AssetManager
        - TaskSystem
        
        Args:
            enable: Whether to include default systems
            
        Returns:
            Self for chaining
        """
        self._use_default_systems = enable
        return self
        
    def add_system(self, system_cls: Type[BaseSystem]):
        """
        Register additional custom system.
        
        Args:
            system_cls: System class to register
            
        Returns:
            Self for chaining
        """
        self._systems.append(system_cls)
        return self
    
    def add_bundle(self, bundle: SystemBundle) -> "ApplicationBuilder":
        """
        Register all systems from a bundle.
        
        Bundles group related systems together, reducing verbosity
        and encapsulating dependency order.
        
        Args:
            bundle: SystemBundle instance to register
            
        Returns:
            Self for chaining
        """
        bundle.register(self)
        return self
        
    def with_logging(self, enable: bool = True):
        """
        Configure logging setup.
        
        Args:
            enable: Whether to setup logging
            
        Returns:
            Self for chaining
        """
        self._logging_configured = enable
        return self
        
    async def build(self) -> ServiceLocator:
        """
        Initialize and start all systems.
        
        Returns:
            ServiceLocator instance with all systems started
        """
        # 0. Load environment variables (Secrets)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.debug("Loaded environment variables from .env")
        except ImportError:
            logger.debug("python-dotenv not installed - skipping .env loading")

        # 1. Setup logging
        if self._logging_configured:
            from .logging import setup_logging
            setup_logging()
            logger.info(f"Starting {self.name}")
        
        # 2. Initialize service locator
        sl.init(self.config_path)
        
        # 3. Register default systems
        if self._use_default_systems:
            from .scheduling import PeriodicTaskScheduler
            
            default_systems = [
                DatabaseManager,
                CommandBus,
                JournalService,
                AssetManager,
                TaskSystem,
                PeriodicTaskScheduler  # NEW: Automated maintenance scheduler
            ]
            for sys_cls in default_systems:
                sl.register_system(sys_cls)
                
        # 4. Register custom systems
        for sys_cls in self._systems:
            sl.register_system(sys_cls)
            
        # 5. Start all systems
        await sl.start_all()
        
        return sl


def run_app(
    main_window_cls,
    viewmodel_cls,
    builder: Optional[ApplicationBuilder] = None,
    app_name: str = "Foundation App",
    config_path: str = "config.json"
):
    """
    One-liner to run a complete Foundation application.
    
    Handles:
    - Qt application setup
    - Event loop configuration
    - Async initialization
    - Main window creation
    - Graceful shutdown
    
    Example:
        from foundation import run_app, ApplicationBuilder
        from .ui.main_window import MainWindow
        from .ui.viewmodels.main_viewmodel import MainViewModel
        from .core.search_service import SearchService
        
        builder = (ApplicationBuilder("My App")
                   .with_default_systems()
                   .add_system(SearchService))
        
        run_app(MainWindow, MainViewModel, builder=builder)
    
    Args:
        main_window_cls: MainWindow class
        viewmodel_cls: ViewModel class for main window
        builder: Optional pre-configured ApplicationBuilder
        app_name: Application name (used if builder not provided)
        config_path: Config path (used if builder not provided)
    """
    
    # Create builder if not provided
    if builder is None:
        builder = ApplicationBuilder(app_name, config_path).with_default_systems()
    
    async def async_main():
        """Async application entry point."""
        # Initialize all systems
        service_locator = await builder.build()
        
        # Create main window
        from ..ui.mvvm.provider import ViewModelProvider
        provider = ViewModelProvider(service_locator)
        main_vm = provider.get(viewmodel_cls)
        
        window = main_window_cls(main_vm)
        window.show()
        
        logger.info(f"{builder.name} started successfully")
        return service_locator
    
    try:
        # Setup Qt application and event loop
        app = QApplication(sys.argv)
        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # Ensure app quits when last window closes
        app.setQuitOnLastWindowClosed(True)
        
        # Stop event loop when app is about to quit
        app.aboutToQuit.connect(loop.stop)
        
        with loop:
            # Run async initialization
            service_locator = loop.run_until_complete(async_main())
            
            # Run event loop (will stop when app.quit() is called)
            loop.run_forever()
            
            # Graceful shutdown
            if service_locator:
                loop.run_until_complete(service_locator.stop_all())
                
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except RuntimeError as e:
        if "Event loop stopped" not in str(e):
            raise
