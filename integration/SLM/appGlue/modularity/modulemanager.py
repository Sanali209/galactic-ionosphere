import importlib

from loguru import logger


class Module:
    def __init__(self):
        self.disabled = False
        self.import_path = None
        self.initialized = False
        self.before_load_module_callbacks = []
        self.after_load_module_callbacks = []
        self.module_settings = {}

    def before_load_module(self, callback):
        self.before_load_module_callbacks.append(callback)

    def after_load_module(self, callback):
        self.after_load_module_callbacks.append(callback)

    def load(self):
        if self.initialized:
            return
        logger.debug(f"module {str(type(self))} - enabled{not self.disabled}")
        if self.disabled:
            return
        if self.import_path is None:
            return
        # before load callbacks
        for callback in self.before_load_module_callbacks:
            callback(self)
        import importlib
        importlib.import_module(self.import_path)
        # after load callbacks
        for callback in self.after_load_module_callbacks:
            callback(self)
        self.initialized = True


class ModuleManager:
    """
    Module manager

    This class is responsible for managing modules.

    Modules are python files that are loaded at runtime and can be enabled or disabled.

    Call the `register` method to register a module.

    Import module file before calling `initialize` method.

    Call the `initialize` method to initialize all registered modules.

    Sample usage:

        ```python
        import ModuleManager, Module
        module_manager = ModuleManager()
        module_manager.initialize()

    """
    known_modules = {}
    before_load_module_callbacks = []
    after_load_module_callbacks = []

    def before_load_modules(self, callback):
        self.before_load_module_callbacks.append(callback)

    def after_load_modules(self, callback):
        self.after_load_module_callbacks.append(callback)

    def register(self):
        def decorator(cls):
            module: Module = cls()
            self.known_modules[type(module)] = module
            return cls

        return decorator

    def get_module(self, module_name):
        return self.known_modules.get(module_name)

    def initialize(self):
        for callback in self.before_load_module_callbacks:
            callback(self)
        for module in self.known_modules.values():
            module.load()
