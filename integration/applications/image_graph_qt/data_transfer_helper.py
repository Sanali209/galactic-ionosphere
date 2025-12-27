import json
import dataclasses
import pathlib
import typing
import threading
import weakref

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtGui import QGuiApplication
from loguru import logger
from bson import ObjectId


# Попытка импортировать MongoRecordWrapper. Обработаем ошибку, если путь неверный.
try:
    from Python.SLM.mongoext.wraper import MongoRecordWrapper
except ImportError as e:
    logger.error(f"Could not import MongoRecordWrapper. Ensure the path is correct in sys.path or adjust the import: {e}")
    # Определим заглушку, чтобы код не падал при запуске, но функциональность Mongo будет недоступна
    class MongoRecordWrapper:
        _id: typing.Any = None
        def __init__(self, oid):
            logger.error("MongoRecordWrapper could not be imported. Mongo transfer functionality is disabled.")
            self._id = oid


# --- Базовый класс для простых данных ---
@dataclasses.dataclass
class DataTClass:
    """Пример базового класса для передачи простых данных."""
    data: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def load_from_json(cls, json_data: typing.Dict[str, typing.Any]):
        """Загружает экземпляр из словаря JSON."""
        # Простая реализация: предполагаем, что json_data - это словарь для атрибута data
        # В подклассах можно переопределить для более сложной логики
        return cls(data=json_data)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        """Сериализует объект в словарь."""
        # Используем стандартный метод dataclasses, если это dataclass
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        return self.data # Возвращаем data для базового класса


# --- Основной класс-хелпер ---
class DataTransferHelper:
    """
    Управляет регистрацией классов, кодированием/декодированием данных
    для drag-and-drop и copy-paste через QMimeData.
    """
    maping: typing.ClassVar[typing.Dict[str, type]] = {}
    _instance = None
    _lock = threading.RLock()

    # MIME типы
    CUSTOM_DATA_MIME_TYPE = 'application/x-custom-data-list'
    MONGO_WRAPPER_MIME_TYPE = 'application/x-mongoextwraper-list'

    def __new__(cls, *args, **kwargs):
        # Реализация синглтона
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # Авто-регистрация базовых типов при первом создании
                logger.debug("Initializing DataTransferHelper singleton.")
                cls.register_class(DataTClass)
                # Зарегистрируем MongoRecordWrapper, если он был успешно импортирован
                if 'MongoRecordWrapper' in globals() and MongoRecordWrapper.__module__ != __name__: # Проверка, что это не заглушка
                     cls.register_class(MongoRecordWrapper)
                else:
                     logger.warning("MongoRecordWrapper was not imported correctly, skipping registration.")
            return cls._instance

    @classmethod
    def register_class(cls, target_cls: type):
        """Регистрирует класс для последующей десериализации."""
        with cls._lock:
            name = target_cls.__name__
            if name in cls.maping and cls.maping[name] != target_cls:
                logger.warning(f"Class name collision: '{name}' already registered for {cls.maping[name]}. Overwriting with {target_cls}.")
            cls.maping[name] = target_cls
            logger.debug(f"Registered class: {name} -> {target_cls}")

    @classmethod
    def get_class_by_name(cls, name: str) -> typing.Optional[type]:
        """Возвращает зарегистрированный класс по имени."""
        return cls.maping.get(name)

    # --- Методы декодирования ---

    def decode(self, mime_data: QMimeData) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """
        Парсит QMimeData и возвращает структурированные данные.
        Возвращает словарь вида {'type': str, 'data': Any} или None.
        """
        if not mime_data:
            logger.warning("Decode called with None mime_data.")
            return None

        formats = mime_data.formats()
        logger.debug(f"Decoding data with formats: {formats}")

        # Приоритеты: Пользовательские -> JSON -> URI-list
        if self.CUSTOM_DATA_MIME_TYPE in formats:
            return self._parse_custom_data_list(mime_data)
        elif self.MONGO_WRAPPER_MIME_TYPE in formats:
             return self._parse_mongo_wrapper_list(mime_data)
        elif 'application/json' in formats: # Попытка разобрать как один из пользовательских
             return self._parse_generic_json(mime_data)
        elif 'text/uri-list' in formats:
            return self._parse_uri_list(mime_data)
        else:
            logger.warning(f"No supported format found in mime data formats: {formats}")
            return None

    def _parse_custom_data_list(self, mime_data: QMimeData) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Парсит список кастомных классов (DataTClass, dataclasses)."""
        logger.debug(f"Parsing '{self.CUSTOM_DATA_MIME_TYPE}'")
        try:
            data_bytes = mime_data.data(self.CUSTOM_DATA_MIME_TYPE)
            # PyQt возвращает QByteArray, нужно получить его данные
            json_list = json.loads(bytes(data_bytes).decode('utf-8'))
            instances = []
            if not isinstance(json_list, list):
                 logger.warning(f"Expected a list in '{self.CUSTOM_DATA_MIME_TYPE}', got {type(json_list)}")
                 return None

            for item in json_list:
                if not isinstance(item, dict):
                    logger.warning(f"Expected dict item in custom data list, got {type(item)}")
                    continue
                class_name = item.get('class')
                class_data = item.get('data')
                if not class_name or class_data is None:
                    logger.warning(f"Invalid item format in custom data list: {item}")
                    continue

                cls = self.get_class_by_name(class_name)
                if not cls:
                    logger.warning(f"Class '{class_name}' not registered.")
                    continue

                try:
                    instance = None
                    # Проверяем наличие метода load_from_json (приоритет)
                    if hasattr(cls, 'load_from_json') and callable(cls.load_from_json):
                         instance = cls.load_from_json(class_data)
                    # Иначе проверяем, является ли он dataclass
                    elif dataclasses.is_dataclass(cls):
                         instance = cls(**class_data) # Стандартный dataclass
                    else:
                         logger.warning(f"Registered class '{class_name}' has no 'load_from_json' and is not a dataclass. Cannot instantiate.")
                         continue
                    instances.append(instance)
                    logger.debug(f"Successfully instantiated {class_name}")
                except Exception as e:
                    logger.error(f"Failed to instantiate class '{class_name}' with data {class_data}: {e}", exc_info=True)

            return {'type': 'custom_data', 'data': instances} if instances else None
        except Exception as e:
            logger.error(f"Failed to parse '{self.CUSTOM_DATA_MIME_TYPE}': {e}", exc_info=True)
            return None

    def _parse_mongo_wrapper_list(self, mime_data: QMimeData) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Парсит список ссылок на MongoRecordWrapper."""
        logger.debug(f"Parsing '{self.MONGO_WRAPPER_MIME_TYPE}'")
        # Проверяем, доступен ли MongoRecordWrapper
        if 'MongoRecordWrapper' not in globals() or MongoRecordWrapper.__module__ == __name__:
             logger.error("MongoRecordWrapper is not available. Cannot parse mongo wrapper list.")
             return None
        try:
            data_bytes = mime_data.data(self.MONGO_WRAPPER_MIME_TYPE)
            json_list = json.loads(bytes(data_bytes).decode('utf-8'))
            instances = []
            if not isinstance(json_list, list):
                 logger.warning(f"Expected a list in '{self.MONGO_WRAPPER_MIME_TYPE}', got {type(json_list)}")
                 return None

            for item in json_list:
                if not isinstance(item, dict):
                    logger.warning(f"Expected dict item in mongo wrapper list, got {type(item)}")
                    continue
                class_name = item.get('class')
                record_id_str = item.get('id')
                if not class_name or not record_id_str:
                    logger.warning(f"Invalid item format in mongo wrapper list: {item}")
                    continue

                cls = self.get_class_by_name(class_name)
                if not cls or not issubclass(cls, MongoRecordWrapper):
                     logger.warning(f"Class '{class_name}' not registered or not a MongoRecordWrapper subclass.")
                     continue

                try:
                    # Используем метакласс MongoRecordWrapper для получения экземпляра (из кеша или БД)
                    instance = cls(record_id_str) # Передаем строку ID
                    instances.append(instance)
                    logger.debug(f"Successfully retrieved instance for {class_name} with id {record_id_str}")
                except Exception as e:
                    logger.error(f"Failed to get instance for {class_name} with id {record_id_str}: {e}", exc_info=True)

            return {'type': 'mongo_wrapper', 'data': instances} if instances else None
        except Exception as e:
            logger.error(f"Failed to parse '{self.MONGO_WRAPPER_MIME_TYPE}': {e}", exc_info=True)
            return None

    def _parse_generic_json(self, mime_data: QMimeData) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Пытается разобрать application/json как один из известных форматов."""
        logger.debug("Attempting to parse generic application/json")
        try:
            data_bytes = mime_data.data('application/json')
            json_string = bytes(data_bytes).decode('utf-8')
            json_data = json.loads(json_string)

            # Простая эвристика: если это список словарей с 'class' и 'id' -> mongo
            if isinstance(json_data, list) and json_data and all(isinstance(i, dict) and 'class' in i and 'id' in i for i in json_data):
                 logger.debug("Detected potential mongo_wrapper list in generic JSON")
                 # Создаем временный QMimeData с правильным типом
                 temp_mime = QMimeData()
                 temp_mime.setData(self.MONGO_WRAPPER_MIME_TYPE, data_bytes)
                 return self._parse_mongo_wrapper_list(temp_mime)
            # Эвристика: если список словарей с 'class' и 'data' -> custom_data
            elif isinstance(json_data, list) and json_data and all(isinstance(i, dict) and 'class' in i and 'data' in i for i in json_data):
                 logger.debug("Detected potential custom_data list in generic JSON")
                 temp_mime = QMimeData()
                 temp_mime.setData(self.CUSTOM_DATA_MIME_TYPE, data_bytes)
                 return self._parse_custom_data_list(temp_mime)
            else:
                logger.warning("Could not determine specific type from generic JSON. Returning as raw data.")
                # Возвращаем как есть, чтобы вызывающий код мог попытаться разобраться
                return {'type': 'generic_json', 'data': json_data}
        except Exception as e:
            logger.error(f"Failed to parse 'application/json': {e}", exc_info=True)
            return None

    def _parse_uri_list(self, mime_data: QMimeData) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Парсит список URI (файлы/папки)."""
        logger.debug("Parsing 'text/uri-list'")
        try:
            urls = mime_data.urls()
            paths = [pathlib.Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
            logger.debug(f"Parsed paths: {paths}")
            return {'type': 'files', 'data': paths} if paths else None
        except Exception as e:
            logger.error(f"Failed to parse 'text/uri-list': {e}", exc_info=True)
            return None

    # --- Методы кодирования (статические) ---

    @staticmethod
    def prepare_file_folder_data(paths: typing.List[typing.Union[str, pathlib.Path]]) -> QMimeData:
        """Готовит QMimeData для списка путей к файлам/папкам."""
        mime_data = QMimeData()
        valid_paths = []
        urls = []
        for p in paths:
            try:
                path = pathlib.Path(p)
                # Проверка существования не обязательна для URL, но полезна для отладки
                # if path.exists():
                urls.append(QUrl.fromLocalFile(str(path.resolve())))
                valid_paths.append(str(path.resolve()))
                # else:
                #    logger.warning(f"Path does not exist, skipping: {path}")
            except Exception as e:
                logger.error(f"Could not create QUrl for path {p}: {e}")

        if urls:
            mime_data.setUrls(urls)
            mime_data.setText(f"Files/Folders: {', '.join(valid_paths)}") # Простой текст для совместимости
            logger.debug(f"Prepared file/folder data for paths: {valid_paths}")
        else:
             logger.warning("No valid paths provided to prepare_file_folder_data.")
        return mime_data

    @staticmethod
    def prepare_mongo_wrapper_data(objects: typing.List[MongoRecordWrapper]) -> QMimeData:
        """Готовит QMimeData для списка экземпляров MongoRecordWrapper."""
        mime_data = QMimeData()
        # Проверяем, доступен ли MongoRecordWrapper
        if 'MongoRecordWrapper' not in globals() or MongoRecordWrapper.__module__ == __name__:
             logger.error("MongoRecordWrapper is not available. Cannot prepare mongo wrapper data.")
             return mime_data

        data_list = []
        valid_objects_count = 0
        for obj in objects:
            if not isinstance(obj, MongoRecordWrapper):
                logger.warning(f"Skipping non-MongoRecordWrapper object: {type(obj)}")
                continue
            class_name = obj.__class__.__name__
            # Убедимся, что класс зарегистрирован
            registered_class = DataTransferHelper.get_class_by_name(class_name)
            if not registered_class or not issubclass(registered_class, MongoRecordWrapper):
                 logger.warning(f"Attempting to transfer unregistered or invalid MongoRecordWrapper subclass: {class_name}. Register it first.")
                 continue # Пропускаем незарегистрированные или невалидные

            if obj._id is None:
                 logger.warning(f"Skipping MongoRecordWrapper object of type {class_name} with None _id.")
                 continue

            data_list.append({'class': class_name, 'id': str(obj._id)})
            valid_objects_count += 1

        if not data_list:
            logger.warning("No valid MongoRecordWrapper objects to prepare for transfer.")
            return mime_data

        try:
            json_string = json.dumps(data_list)
            mime_data.setData(DataTransferHelper.MONGO_WRAPPER_MIME_TYPE, json_string.encode('utf-8'))
            mime_data.setText(f"MongoWrapper List ({valid_objects_count} items)") # Для простого текста
            logger.debug(f"Prepared mongo wrapper data ({valid_objects_count} items): {json_string}")
        except Exception as e:
            logger.error(f"Failed to serialize mongo wrapper data: {e}", exc_info=True)

        return mime_data

    @staticmethod
    def prepare_custom_class_data(objects: typing.List[typing.Any]) -> QMimeData:
        """Готовит QMimeData для списка экземпляров кастомных классов (DataTClass, dataclasses)."""
        mime_data = QMimeData()
        data_list = []
        valid_objects_count = 0
        for obj in objects:
            cls = obj.__class__
            class_name = cls.__name__

            # Убедимся, что класс зарегистрирован
            registered_class = DataTransferHelper.get_class_by_name(class_name)
            if not registered_class:
                 logger.warning(f"Attempting to transfer unregistered class: {class_name}. Register it first.")
                 continue # Пропускаем

            try:
                data_dict = None
                if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                    data_dict = obj.to_dict()
                elif dataclasses.is_dataclass(obj):
                    data_dict = dataclasses.asdict(obj)
                else:
                    logger.warning(f"Cannot serialize object of type {class_name}. No 'to_dict' method and not a dataclass.")
                    continue

                if data_dict is not None:
                    data_list.append({'class': class_name, 'data': data_dict})
                    valid_objects_count += 1
            except Exception as e:
                 logger.error(f"Failed to serialize object of class {class_name}: {e}", exc_info=True)

        if not data_list:
            logger.warning("No valid custom class objects to prepare for transfer.")
            return mime_data

        try:
            json_string = json.dumps(data_list)
            mime_data.setData(DataTransferHelper.CUSTOM_DATA_MIME_TYPE, json_string.encode('utf-8'))
            mime_data.setText(f"Custom Data List ({valid_objects_count} items)")
            logger.debug(f"Prepared custom class data ({valid_objects_count} items): {json_string}")
        except Exception as e:
            logger.error(f"Failed to serialize custom class data: {e}", exc_info=True)

        return mime_data

    # --- Методы для буфера обмена ---

    def copy_to_clipboard(self, mime_data: QMimeData):
        """Копирует предоставленные QMimeData в системный буфер обмена."""
        if not mime_data or not mime_data.formats():
            logger.warning("Attempted to copy empty or invalid mime_data to clipboard.")
            return
        try:
            clipboard = QGuiApplication.clipboard()
            if clipboard:
                clipboard.setMimeData(mime_data)
                logger.info(f"Copied data with formats {mime_data.formats()} to clipboard.")
            else:
                logger.error("Could not get system clipboard.")
        except Exception as e:
            logger.error(f"Failed to copy data to clipboard: {e}", exc_info=True)

    def get_from_clipboard(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Извлекает и декодирует данные из системного буфера обмена."""
        logger.info("Attempting to get data from clipboard.")
        try:
            clipboard = QGuiApplication.clipboard()
            if not clipboard:
                logger.error("Could not get system clipboard.")
                return None

            mime_data = clipboard.mimeData()
            if not mime_data or not mime_data.formats():
                logger.info("Clipboard is empty or contains no supported formats.")
                return None

            return self.decode(mime_data)
        except Exception as e:
            logger.error(f"Failed to get or decode data from clipboard: {e}", exc_info=True)
            return None


# --- Функция инициализации ---
def initialize_data_transfer():
    """
    Инициализирует DataTransferHelper (создает синглтон) и
    регистрирует необходимые классы приложения.
    Вызывать один раз при старте приложения.
    """
    logger.info("Initializing Data Transfer subsystem...")
    helper = DataTransferHelper() # Создаст синглтон и зарегистрирует базовые классы

    # !!! ВАЖНО: Зарегистрируйте здесь все ваши подклассы MongoRecordWrapper
    # и другие кастомные классы (DataTClass, dataclasses), которые
    # предполагается передавать через drag-and-drop или copy-paste. !!!

    # Пример:
    # try:
    #     from Python.applications.image_graph_qt2.models import FileRecord, TagRecord # Пример пути
    #     helper.register_class(FileRecord)
    #     helper.register_class(TagRecord)
    #     logger.info("Registered FileRecord and TagRecord.")
    # except ImportError:
    #     logger.error("Could not import FileRecord or TagRecord for registration.")

    # try:
    #     from my_other_module import MyCustomDataclass
    #     helper.register_class(MyCustomDataclass)
    #     logger.info("Registered MyCustomDataclass.")
    # except ImportError:
    #      logger.warning("Could not import MyCustomDataclass for registration.")

    logger.info("DataTransferHelper initialized. Ensure all necessary classes are registered.")


# Пример использования (для тестирования):
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    # Нужен QApplication для доступа к буферу обмена
    app = QApplication(sys.argv)

    logger.info("Running DataTransferHelper example...")
    initialize_data_transfer()
    helper = DataTransferHelper()

    # --- Тест 1: Файлы ---
    print("\n--- Testing File Transfer ---")
    # Создадим временные файлы для теста
    temp_dir = pathlib.Path("./temp_transfer_test")
    temp_dir.mkdir(exist_ok=True)
    file1 = temp_dir / "test1.txt"
    file2 = temp_dir / "test2.log"
    file1.write_text("Hello")
    file2.write_text("World")
    paths_to_copy = [file1, temp_dir]

    mime_files = helper.prepare_file_folder_data(paths_to_copy)
    print(f"Prepared File MIME formats: {mime_files.formats()}")
    print(f"Prepared File MIME urls: {[url.toLocalFile() for url in mime_files.urls()]}")

    helper.copy_to_clipboard(mime_files)
    decoded_files = helper.get_from_clipboard()
    print(f"Decoded from clipboard (Files): {decoded_files}")

    # Очистка временных файлов
    file1.unlink()
    file2.unlink()
    temp_dir.rmdir()

    # --- Тест 2: Кастомные классы ---
    print("\n--- Testing Custom Class Transfer ---")
    @dataclasses.dataclass
    class MyData(DataTClass): # Наследуемся от DataTClass или просто dataclass
        name: str = "default"
        value: int = 0

        # Не обязательно, но можно определить to_dict, если нужна особая логика
        # def to_dict(self): return {"name": self.name, "value": self.value}

        # Не обязательно, но можно определить load_from_json
        # @classmethod
        # def load_from_json(cls, json_data): return cls(name=json_data.get("name", ""), value=json_data.get("value", 0))


    helper.register_class(MyData) # Регистрируем наш класс

    obj1 = MyData(name="Test1", value=123)
    obj2 = DataTClass(data={"key": "value"}) # Используем базовый DataTClass

    mime_custom = helper.prepare_custom_class_data([obj1, obj2])
    print(f"Prepared Custom MIME formats: {mime_custom.formats()}")
    print(f"Prepared Custom MIME data: {bytes(mime_custom.data(helper.CUSTOM_DATA_MIME_TYPE)).decode('utf-8')}")

    helper.copy_to_clipboard(mime_custom)
    decoded_custom = helper.get_from_clipboard()
    print(f"Decoded from clipboard (Custom): {decoded_custom}")
    if decoded_custom and decoded_custom['type'] == 'custom_data':
        for item in decoded_custom['data']:
            print(f"  - Decoded item: {item} (Type: {type(item)})")


    # --- Тест 3: Mongo Wrapper (Заглушка, если MongoRecordWrapper не импортирован) ---
    print("\n--- Testing Mongo Wrapper Transfer ---")
    if 'MongoRecordWrapper' in globals() and MongoRecordWrapper.__module__ != __name__:
        # Этот блок выполнится только если MongoRecordWrapper был реально импортирован
        # Вам нужно будет создать реальные экземпляры ваших подклассов MongoRecordWrapper
        # Например:
        # mongo_obj1 = FileRecord(ObjectId()) # Предполагая, что FileRecord зарегистрирован
        # mongo_obj2 = TagRecord(ObjectId())  # Предполагая, что TagRecord зарегистрирован
        # mime_mongo = helper.prepare_mongo_wrapper_data([mongo_obj1, mongo_obj2])
        # ... (дальнейшее тестирование как с custom классами) ...
        print("MongoRecordWrapper seems imported. Add specific test cases with your subclasses.")
        # Пример с заглушкой, если реального импорта нет, но класс определен
        class MockMongoSubclass(MongoRecordWrapper): pass
        helper.register_class(MockMongoSubclass)
        mock_mongo1 = MockMongoSubclass(ObjectId())
        mock_mongo2 = MockMongoSubclass(ObjectId())
        mime_mongo = helper.prepare_mongo_wrapper_data([mock_mongo1, mock_mongo2])
        print(f"Prepared Mongo MIME formats: {mime_mongo.formats()}")
        if mime_mongo.hasFormat(helper.MONGO_WRAPPER_MIME_TYPE):
             print(f"Prepared Mongo MIME data: {bytes(mime_mongo.data(helper.MONGO_WRAPPER_MIME_TYPE)).decode('utf-8')}")
        helper.copy_to_clipboard(mime_mongo)
        decoded_mongo = helper.get_from_clipboard()
        print(f"Decoded from clipboard (Mongo): {decoded_mongo}")
        if decoded_mongo and decoded_mongo['type'] == 'mongo_wrapper':
            for item in decoded_mongo['data']:
                print(f"  - Decoded item ID: {item._id} (Type: {type(item)})")

    else:
        print("MongoRecordWrapper not available or is a stub. Skipping detailed Mongo transfer test.")

    print("\nExample finished.")
    # sys.exit(app.exec()) # Не нужно для скрипта, только для GUI приложения
