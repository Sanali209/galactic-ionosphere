from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp


class AttributeParser:
    """
    A base class for parsing a named attribute out of a dictionary and applying it to a widget.
    """

    def __init__(self, name: str, value_type: type = None, required: bool = False):
        self.name = name
        self.value_type = value_type
        self.required = required
        self.value = None

    def is_exist(self, element_parser) -> bool:
        """
        Checks if the attribute name is present in the widget's data dictionary.
        Raises ValueError if the attribute is required but not present.
        """
        data = element_parser.widget_dictionary
        exists = self.name in data
        if self.required and not exists:
            raise ValueError(f"Attribute '{self.name}' is required but not found in {data}")
        return exists

    def get_value(self, element_parser):
        """
        Retrieves the raw value from the dictionary and optionally casts it to a given type.
        """
        self.value = element_parser.widget_dictionary[self.name]
        if self.value_type is not None:
            self.value = self.value_type(self.value)
        else:
            self.value = self.value

    def parse(self, element_parser):
        """
        If the attribute exists, fetch the value and then set it on the widget.
        """
        if self.is_exist(element_parser):
            self.get_value(element_parser)
            self.set_element_attribute(element_parser)

    def set_element_attribute(self, element_parser):
        """
        Applies the parsed attribute value to the widget. By default uses Python's setattr.
        Override if you need custom logic.
        """
        if self.value is not None:
            setattr(element_parser._widget, self.name, self.value)


class IdAttributeParser(AttributeParser):
    """
    Example of a specialized attribute parser that handles an 'id' attribute.
    """

    def __init__(self):
        # By default, just parse 'id' as a string (not strictly required)
        super().__init__("id", str)

    def set_element_attribute(self, element_parser):
        """
        Stores the 'id' attribute on the widget itself. You could also
        register this ID somewhere else if you like.
        """
        if self.value is not None:
            setattr(element_parser._widget, "id", self.value)

class FixedHeightAttribute(AttributeParser):
    def __init__(self):
        # By default, just parse 'id' as a string (not strictly required)
        super().__init__("FixedHeight", int)

    def set_element_attribute(self, element_parser):
        if self.value is not None:
            element_parser._widget.setFixedHeight(self.value)


class Button_on_clickAttribute(AttributeParser):
    """
    Example of a specialized attribute parser that handles an 'on_click' attribute.
    """

    def __init__(self):
        # By default, just parse 'on_click' as a string (not strictly required)
        super().__init__(name="on_click", required=False)

    def set_element_attribute(self, element_parser):
        """
        Stores the 'on_click' attribute on the widget itself. You could also
        register this ID somewhere else if you like.
        """
        if self.value is not None:
            element_parser._widget.clicked.connect(self.value)


class setTextAttribute(AttributeParser):
    """
    Example of a specialized attribute parser that handles an 'on_click' attribute.
    """

    def __init__(self, name):
        # By default, just parse 'on_click' as a string (not strictly required)
        super().__init__(name=name, required=False)

    def set_element_attribute(self, element_parser):
        """
        Stores the 'on_click' attribute on the widget itself. You could also
        register this ID somewhere else if you like.
        """
        if self.value is not None:
            element_parser._widget.setText(self.value)


class ElementParser:
    """
    The main parser responsible for:
      - Storing the data dictionary
      - Creating/reusing the widget
      - Parsing all attributes
      - Handling layout definitions (if any)
      - Recursively parsing child widgets
    """

    def __init__(self, parent_parser=None):
        self.parent_parser = parent_parser
        self._widget = None
        self.root_widget = None

        # Will hold the dictionary for this specific widget
        self.widget_dictionary = {}

        # Register here whichever attribute parsers you want to apply
        self.attributes = [
            IdAttributeParser(),
            FixedHeightAttribute(),
            # Potentially add more: e.g., TextAttributeParser, etc.
        ]

        # Keep track of child element parsers if needed
        self.children_parsers = []
        self.parsers_types = {QPushButton: QButtonParser, QLabel: QLabelParser}

    def parse(self, dict_data) -> QWidget:
        """
        Main entry point. Expects a dictionary with a least a "type" or "instance".
        Example:
            {
                "type": QWidget,
                "layout": {
                    QVBoxLayout: [
                        {"type": QLabel, "text": "Hello", "key": "my_label"}
                    ]
                }
            }
        """
        self.widget_dictionary = dict_data

        widget_type = dict_data.get("type")
        pred_instance = dict_data.get("instance")

        # 1) Create/reuse the widget
        if pred_instance is not None:
            self._widget = pred_instance
        else:
            if widget_type is not None and callable(widget_type):
                self._widget = widget_type()
            else:
                raise ValueError("No widget 'type' callable or 'instance' provided.")

        parsed_atr = []
        # 2) Parse known attributes using AttributeParser objects
        for attr_parser in self.attributes:
            attr_parser.parse(self)
            parsed_atr.append(attr_parser)

        #3) parse attached attributes expect list of dictionaries with key and value
        attached = dict_data.get("attached", [])
        for attach in attached:
            value = attach.get("value")
            key = attach.get("key")
            setattr(self._widget, key, value)

        # handle "menu_bar" attribute
        menu_bar_data = dict_data.get("menu_bar")
        if isinstance(self._widget, QWindow) and menu_bar_data:
            self._widget: QWindow
            menu_bar =self._widget.menuBar()
        else:
            try:
                menu_bar = self._widget._main_window.menuBar()
            except:
                menu_bar = None
        if menu_bar:
            for menus in menu_bar_data:
                menu = menu_bar.addMenu(menus.get("title"))
                for menu_items in menus.get("items",[]):
                    title = menu_items.get("title")
                    triggered = menu_items.get("triggered")
                    menu.addAction(title, triggered)



        # 3) Parse layout (if present)
        layout_def = dict_data.get("layout", None)
        if layout_def and isinstance(layout_def, dict):
            # Typically we expect a dictionary with { LayoutClass: [list of child definitions] }
            layout_keys = layout_def.keys()
            first_key = next(iter(layout_keys))

            layout_class = first_key
            children_list = layout_def.get(layout_class)
            layout = layout_class()
            self._widget.setLayout(layout)
            setContentsMargins = dict_data.get("setContentsMargins", None)
            if setContentsMargins:
                layout.setContentsMargins(*setContentsMargins)
            setFixedSize = dict_data.get("setFixedSize", None)
            if setFixedSize:
                self._widget.setFixedSize(*setFixedSize)
            setFixedHeight = dict_data.get("FixedHeight", None)
            if setFixedHeight:
                self._widget.setFixedHeight(setFixedHeight)

            # For each child dict, recursively parse
            for child_data in children_list:
                child_parser = self.parsers_types.get(child_data["type"], ElementParser)(self)
                child_widget = child_parser.parse(child_data)
                self.children_parsers.append(child_parser)

                # If there's a "key", set it as an attribute on the parent widget
                child_key = child_data.get("key")
                if child_key:
                    setattr(self._widget, child_key, child_widget)

                layout.addWidget(child_widget)

        # Expose whatever widget we built as root_widget
        self.root_widget = self._widget
        return self._widget


class QButtonParser(ElementParser):
    """
    A specialized parser for QPushButton widgets.
    """

    def __init__(self, parent_parser=None):
        super().__init__(parent_parser)
        self.attributes = [
            IdAttributeParser(),
            setTextAttribute("text"),
            Button_on_clickAttribute(),
        ]

class QLabelParser(ElementParser):
    """
    A specialized parser for QLabel widgets.
    """

    def __init__(self, parent_parser=None):
        super().__init__(parent_parser)
        self.attributes = [
            IdAttributeParser(),
            setTextAttribute("text"),
        ]


if __name__ == "__main__":
    QtApp = PySide6GlueApp()
    given_instance = QWidget()
    QtApp.set_main_widget(given_instance)
    # Example usage:
    parser = ElementParser()

    widget_data = {
        # You can either pass an existing instance or a type to construct
        "type": QWidget,
        "instance": given_instance,

        # Example layout definition:
        "layout": {
            QVBoxLayout: [
                {
                    "type": QLabel,
                    "setText": "Hello World",
                    "key": "hello_label"
                },
                {
                    "type": QPushButton,
                    "text": "Click Me",
                    "key": "hello_button",
                    "on_click": lambda: given_instance.hello_label.setText('Hello World 4')
                },
                {
                    "type": QLabel,
                    "text": "Hello World 2",
                    "key": "hello_label2"
                },
            ]
        }
    }

    # Parse to get the (parent) widget
    widget: QWidget = parser.parse(widget_data)

    # You can now access child widgets by their "key" property as attributes:
    widget.hello_label.setText("Hello World 3")

    QtApp.run()
