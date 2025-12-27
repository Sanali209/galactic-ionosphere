
# Документация для core.py

## Содержание
1. [Введение](#введение)
2. [Классы](#классы)
   - [ModeManager](#appcontext)
   - [ContextMode](#contextmode)
   - [Event](#event)
   - [ConfigStorage](#configstorage)
   - [BaseConfig](#baseconfig)
   - [ConfigSection](#configsection)
   - [TypedConfigSection](#typedconfigsection)
   - [ConfigListener](#configlistener)
   - [Allocator](#allocator)
   - [Module](#module)
   - [Resource](#resource)
   - [Service](#service)
   - [ResourcesRepo](#resourcesrepo)
   - [GlueApp](#glueapp)
3. [Примеры использования](#примеры-использования)

## Введение
Этот файл содержит описание основных классов и их функциональности, используемых в `core.py`. Он включает управление конфигурацией, событиями, ресурсами и модулями приложения.

## Классы

### ModeManager
Класс для управления режимами приложения.

### ContextMode
Базовый класс для режимов приложения.

### Event
Класс для управления событиями с подписчиками.

### ConfigStorage
Класс для работы с хранилищем конфигурации.

### BaseConfig
Базовый класс для управления конфигурацией.

### ConfigSection
Класс для представления секции конфигурации.

### TypedConfigSection
Расширенный класс секции конфигурации с предопределенными атрибутами.

### ConfigListener
Слушатель изменений конфигурации.

### Allocator
Класс для управления службами и их инициализацией.

### Module
Класс для представления модуля приложения.

### Resource
Базовый класс для ресурсов приложения.

### Service
Класс для представления службы приложения.

### ResourcesRepo
Класс для управления репозиторием ресурсов.

### GlueApp
Главный класс приложения, управляющий модулями, ресурсами и конфигурацией.

## Примеры использования
Пример инициализации приложения:
```python
app = GlueApp()
app.add_module(Module("ExampleModule", []))
app.init()
app.Run()
```
