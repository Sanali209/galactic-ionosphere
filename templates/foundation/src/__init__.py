"""
Foundation - Desktop Application Framework

A professional framework for building async Python desktop applications
with PySide6, MongoDB, and advanced GUI features.
"""

# Core systems
from src.core.base_system import BaseSystem
from src.core.locator import ServiceLocator, sl
from src.core.config import (
    ConfigManager, 
    AppConfig, 
    AISettings, 
    MongoSettings, 
    GeneralSettings
)
from src.core.events import ObserverEvent
from src.core.logging import setup_logging
from src.core.bootstrap import ApplicationBuilder, run_app

# Database
from src.core.database.manager import DatabaseManager
from src.core.database.orm import (
    CollectionRecord,
    Field,
    StringField,
    IntField,
    BoolField,
    DictField,
    ReferenceField,
    EmbeddedField,
    ListField,
    Reference,
    ReferenceList,
)

# Systems
from src.core.commands.bus import CommandBus
from src.core.journal.service import JournalService
from src.core.assets.manager import AssetManager
from src.core.tasks.system import TaskSystem

# UI Framework
from src.ui.main_window import MainWindow
from src.ui.mvvm.provider import ViewModelProvider
from src.ui.mvvm.viewmodel import BaseViewModel, BindableBase, BindableProperty
from src.ui.mvvm.binding import bind, bind_command, BindingMode
from src.ui.mvvm.data_context import DataContextMixin, BindableWidget
from src.ui.docking.dock_manager import DockManager
from src.ui.docking.panel_base import BasePanelWidget
from src.ui.menus.action_registry import ActionRegistry
from src.ui.menus.menu_builder import MenuBuilder

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseSystem",
    "ServiceLocator",
    "sl",
    "ConfigManager",
    "AppConfig",
    "AISettings",
    "MongoSettings",
    "GeneralSettings",
    "ObserverEvent",
    "setup_logging",
    "ApplicationBuilder",
    "run_app",
    
    # Database
    "DatabaseManager",
    "CollectionRecord",
    "Field",
    "StringField",
    "IntField",
    "BoolField",
    "DictField",
    "ReferenceField",
    "EmbeddedField",
    "ListField",
    "Reference",
    "ReferenceList",
    
    # Systems
    "CommandBus",
    "JournalService",
    "AssetManager",
    "TaskSystem",
    
    # UI
    "MainWindow",
    "ViewModelProvider",
    "BaseViewModel",
    "BindableBase",
    "BindableProperty",
    "bind",
    "bind_command",
    "BindingMode",
    "DataContextMixin",
    "BindableWidget",
    "DockManager",
    "BasePanelWidget",
    "ActionRegistry",
    "MenuBuilder",
]
