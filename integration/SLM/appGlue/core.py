from SLM.appGlue.modularity.modulemanager import ModuleManager

APP_DATA_DIR = r"D:\data"  #os.path.dirname(sys.argv[0])

from typing import TypeVar, Type, List
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem


class ModeManager:
    def __init__(self):
        self.modes = {}
        self.current_mode = None

    def get_mode(self, mode_name):
        return self.modes.get(mode_name)

    def set_mode(self, mode_name):
        mode = self.get_mode(mode_name)
        if mode:
            if self.current_mode is not None:
                self.current_mode.deactivate()
            self.current_mode = mode
            self.current_mode.activate()

    def register_mode(self, mode_name, mode_class):
        """Register a new mode with its class."""
        if not issubclass(mode_class, ContextMode):
            raise TypeError("Mode class must inherit from ContextMode")
        self.modes[mode_name] = mode_class()
        if self.current_mode is None:
            self.set_mode(mode_name)


class ContextMode:
    def activate(self):
        pass

    def deactivate(self):
        pass


class Event:
    def __init__(self):
        self.callbacks = []
        self.enabled = True

    def subscribe(self, callback):
        self.callbacks.append(callback)

    def unsubscribe(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def fire(self, *args, **kwargs):
        if not self.enabled:
            return
        for callback in self.callbacks:
            callback(*args, **kwargs)

    # ovveride __iadd__ to make += work
    def __iadd__(self, callback):
        self.subscribe(callback)
        return self

    # ovveride __isub__ to make -= work
    def __isub__(self, callback):
        self.unsubscribe(callback)
        return self


class ConfigStorage:
    """Класс для работы с хранилищем конфигурации."""

    def __init__(self):
        self.storage = {}

    def load(self):
        """Загружает конфигурацию из хранилища."""
        return self.storage

    def save(self, section_name, attr_name, value):
        """Сохраняет значение в хранилище."""
        if section_name not in self.storage:
            self.storage[section_name] = {}
        self.storage[section_name][attr_name] = value


#todo:refactor to module level
# Define a TypeVar that is bounded by Service
T = TypeVar('T', bound='Service')


class BaseConfig:
    """Базовый класс конфигурации."""

    def __init__(self):
        self._sections = {}
        self._storage = ConfigStorage()
        self.default = {}

    def __getattr__(self, section_name):
        if section_name.startswith("_"):
            return self.__dict__[section_name]
        """Автоматически создает секцию или возвращает зарегистрированную."""
        if section_name not in self._sections:
            self._sections[section_name] = self._create_section(section_name)
        return self._sections[section_name]

    def _create_section(self, section_name):
        """Создает секцию: зарегистрированную или базовую."""
        return ConfigSection(section_name, self)

    def set_storage(self, storage):
        self._storage = storage

    def register_section(self, section_name, section_class):
        """Регистрирует секцию с ее классом."""
        if not issubclass(section_class, ConfigSection):
            raise TypeError("Section class must inherit from ConfigSection")
        self._sections[section_name] = section_class(section_name, self)

    def load_config(self):
        """Загружает конфигурацию из хранилища и уведомляет слушателей."""
        loaded_config = self._storage.load()
        temp_stor = self._storage
        self._storage = ConfigStorage()
        for section_name, section_data in loaded_config.items():
            if section_name not in self._sections:
                section = self._create_section(section_name)
            else:
                section = self._sections[section_name]
            for attr_name, value in section_data.items():
                cur_val = section.__getattr__(attr_name)
                if cur_val != value:
                    section.__setattr__(attr_name, value)
                    #self.notify_change(section_name, attr_name, value)
        self._storage = temp_stor

    def notify_change(self, section_name, attr_name, value):
        """Оповещает слушателей об изменении атрибута секции и сохраняет значение."""
        self._storage.save(section_name, attr_name, value)
        MessageSystem.SendMessage(
            "ConfigChanged",
            section=section_name,
            attribute=attr_name,
            value=value,
        )


class ConfigSection:
    """Базовый класс для секции конфигурации."""

    def __init__(self, name, parent):
        self._name = name
        self._parent = parent
        self._data = {}

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return self.__dict__[attr]
        if attr in self._data:
            return self._data[attr]
        raise AttributeError(f"'{self._name}' section has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr in ["_name", "_parent", "_data"]:
            super().__setattr__(attr, value)
        else:
            self._data[attr] = value
            self._parent.notify_change(self._name, attr, value)


class TypedConfigSection(ConfigSection):
    """Расширенный класс секции с поддержкой предопределенных атрибутов."""

    def __init__(self, name, parent):
        super().__init__(name, parent)
        for attr_name, attr_value in self.__class__.__dict__.items():
            if not attr_name.startswith("_"):
                self._data[attr_name] = attr_value

    def __setattr__(self, attr, value):
        if attr in self.__class__.__dict__:
            setattr(self.__class__, attr, value)
        super().__setattr__(attr, value)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return self.__dict__[attr]
        if attr in self._data:
            return self._data[attr]
        raise AttributeError(f"'{self._name}' section has no attribute '{attr}'")


class ConfigListener:
    """Слушатель изменений конфигурации."""

    def __init__(self):
        MessageSystem.Subscribe("ConfigChanged", self, self.on_config_change)

    def on_config_change(self, section, attribute, value):
        """Обрабатывает изменение атрибута в секции.
        important: name of attributes must be section, attribute, value
        """

        print(f"Изменение: секция={section}, атрибут={attribute}, значение={value}")

    def __del__(self):
        MessageSystem.Unsubscribe("ConfigChanged", self)


class TestConfigStorage(ConfigStorage):
    def __init__(self):
        super().__init__()
        self.storage = {
            "database": {
                "host": "test_host",
                "port": 5432,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            }
        }

    def save(self, section_name, attr_name, value):
        print(f"Сохранение: секция={section_name}, атрибут={attr_name}, значение={value}")
        super().save(section_name, attr_name, value)


class Resource:

    def __init__(self):
        self.name = __class__.__name__
        self.tags = []


class ResourcesRepo:

    def __init__(self):
        self.resources: List[Resource] = []
        self.type_res = {}

    def has_service(self, _type: Type[T]) -> bool:
        if _type in self.type_res.keys(): return True
        return False

    def get_by_type_one(self, _type: Type[T]) -> T:
        try:
            return self.type_res[_type][0]
        except IndexError:
            raise Exception(f'No resources of type {_type} found')

    def get_by_name_one(self, name: str) -> Resource:
        for resource in self.resources:
            if resource.name == name:
                return resource
        raise Exception(f'No resources with name {name} found')

    def register(self, resource: Resource):
        self.resources.append(resource)
        res_type = type(resource)
        list_of_typed_res = self.type_res.get(res_type, [])
        list_of_typed_res.append(resource)
        self.type_res[res_type] = list_of_typed_res

    def register_or_override(self, _type, instance: Resource):
        for i, resource in enumerate(self.resources):
            if isinstance(resource, _type):
                self.resources[i] = instance
                self.type_res[_type] = [instance]
                return


class Module:

    def __init__(self, name: str, dependencies: List[str] = None):
        self.name = name
        self.dependencies = dependencies
        self.enabled = True
        self.initialized = False
        self.loaded = False

    def init(self):
        """define services and register them in Allocator"""
        print(f"Initializing module {self.name}")
        self.initialized = True

    def load(self):
        """additional loading of module resources"""
        print(f"Loading module {self.name}")
        self.loaded = True

    def unload(self):
        """in this place module unload them recources for use in app"""
        print(f"Unloading module {self.name}")
        self.loaded = False


class Allocator:
    config = BaseConfig()
    modules = []
    res: ResourcesRepo = ResourcesRepo()

    @classmethod
    def add_module(cls, module: Module):
        if not isinstance(module, Module):
            raise TypeError("Module must inherit from Module")
        cls.modules.append(module)

    @classmethod
    def disable_module(cls, name: str, value: bool = False):
        for module in cls.modules:
            if module.name == name:
                module.enabled = value
                return
        raise ValueError(f"Module with name {name} not found")

    @classmethod
    def get_instance(cls, key: Type[T]) -> T:
        obj = cls.res.get_by_type_one(key)
        if isinstance(obj, Service):
            if not obj.lasy_init:
                obj.lasy_init = True
                obj.lasy_load()
        return obj

    @classmethod
    def init_modules(cls):
        for module in cls.modules:
            if module.enabled:
                module.init()
        cls.init_services()
        for module in cls.modules:
            if module.enabled:
                module.load()

    @classmethod
    def init_services(cls):
        Allocator.config.load_config()

        for service in Allocator.res.resources:
            if isinstance(service, Service):
                service.init(Allocator.config)

    @classmethod
    def dispose_services(cls):
        for service in Allocator.res.resources:
            if isinstance(service, Service):
                service.dispose()

        for module in Allocator.modules:
            if module.enabled:
                module.unload()
        Allocator.config.save_config()


class Service(Resource):
    def __init__(self):
        super().__init__()
        self.lasy_init = False

    def init(self, config):
        pass

    def dispose(self):
        pass

    def lasy_load(self):
        pass

    @classmethod
    def instance(cls: Type[T]) -> T:
        return Allocator.get_instance(cls)


class ServiceBackend:
    name = "default"
    lazy_loaded = False
    tags = []

    def is_compatible(self, *args, **kwargs):
        return True

    def load(self, *args, **kwargs):
        pass


class BackendProvider(Resource):
    name = "default"

    def __init__(self):
        super().__init__()
        self.backends = {}

    def get_all_backends(self):
        return list(self.backends.keys())

    def get_all_by_tag(self, tag: str):
        compactible = []
        for backend in self.backends.values():
            if tag in backend.tags:
                compactible.append(backend.format)
        return compactible

    def register_backend(self, backend: ServiceBackend):
        if not isinstance(backend, ServiceBackend):
            raise TypeError("Backend must inherit from ServiceBackend")
        self.backends[backend.name] = backend

    def get_backend_by_name(self, name: str) -> ServiceBackend:
        backend: ServiceBackend = self.backends.get(name)
        if backend is None:
            raise ValueError(f"Backend with name {name} not found")
        if not backend.lazy_loaded:
            backend.load()
            backend.lazy_loaded = True
        return backend


class GlueApp(Resource):
    current_app = None
    """current app instance"""

    def __init__(self):
        super().__init__()
        self.config: BaseConfig = BaseConfig()
        GlueApp.current_app = self
        Allocator.res.register(self)

        #self.action_manager: ActionManager = ActionManager()
        self.init()

    def init(self):
        """Initialize the application."""
        Allocator.init_modules()

    def exit(self):
        """Exit the application."""
        Allocator.dispose_services()
        Allocator.config.save_config()

    def run(self):
        pass
