
# Документация пользователя для `parser.py`

## Описание
`parser.py` предоставляет инструменты для создания и настройки виджетов PySide6 на основе словарей данных. Основные компоненты включают классы для парсинга атрибутов (`AttributeParser`) и элементов (`ElementParser`), а также специализированные парсеры для различных виджетов.

---

## Основные классы

### 1. `AttributeParser`
Базовый класс для обработки атрибутов виджетов.

- **Конструктор**:
  ```python
  AttributeParser(name: str, value_type: type = None, required: bool = False)
  ```
  - `name`: Имя атрибута.
  - `value_type`: Тип значения атрибута (опционально).
  - `required`: Указывает, является ли атрибут обязательным.

- **Методы**:
  - `is_exist(element_parser)`: Проверяет наличие атрибута в словаре данных.
  - `get_value(element_parser)`: Извлекает и преобразует значение атрибута.
  - `parse(element_parser)`: Основной метод для обработки атрибута.
  - `set_element_attribute(element_parser)`: Применяет значение атрибута к виджету.

---

### 2. `ElementParser`
Основной класс для парсинга виджетов и их атрибутов.

- **Конструктор**:
  ```python
  ElementParser(parent_parser=None)
  ```
  - `parent_parser`: Родительский парсер (если есть).

- **Методы**:
  - `parse(dict_data)`: Основной метод для создания и настройки виджета на основе словаря данных.
  - Обрабатывает:
    - Тип виджета (`type`) или существующий экземпляр (`instance`).
    - Атрибуты через зарегистрированные парсеры.
    - Макеты (`layout`) и дочерние виджеты.
    - Дополнительные атрибуты (`attached`) и меню (`menu_bar`).

---

### 3. Специализированные парсеры
- `IdAttributeParser`: Обрабатывает атрибут `id`.
- `Button_on_clickAttribute`: Обрабатывает атрибут `on_click` для кнопок.
- `setTextAttribute`: Обрабатывает текстовые атрибуты, такие как `text`.
- `QButtonParser`: Парсер для `QPushButton`.
- `QLabelParser`: Парсер для `QLabel`.

---

## Пример использования

```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from parser import ElementParser

# Пример данных для парсинга
widget_data = {
    "type": QWidget,
    "layout": {
        QVBoxLayout: [
            {
                "type": QLabel,
                "text": "Привет, мир!",
                "key": "label1"
            },
            {
                "type": QPushButton,
                "text": "Нажми меня",
                "key": "button1",
                "on_click": lambda: print("Кнопка нажата!")
            }
        ]
    }
}

# Создание и настройка виджета
parser = ElementParser()
widget = parser.parse(widget_data)

# Доступ к дочерним виджетам через их ключи
widget.label1.setText("Обновленный текст")
```

---

## Расширение
Вы можете добавлять собственные парсеры атрибутов или виджетов, наследуясь от `AttributeParser` или `ElementParser`.

Пример создания нового парсера атрибута:
```python
class CustomAttributeParser(AttributeParser):
    def __init__(self):
        super().__init__(name="custom_attribute", value_type=str)

    def set_element_attribute(self, element_parser):
        if self.value is not None:
            print(f"Применение атрибута: {self.value}")
```

Пример добавления нового парсера виджета:
```python
class CustomWidgetParser(ElementParser):
    def __init__(self, parent_parser=None):
        super().__init__(parent_parser)
        self.attributes.append(CustomAttributeParser())
```

---
