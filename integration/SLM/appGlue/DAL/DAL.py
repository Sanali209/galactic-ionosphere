from typing import Callable, Optional

from SLM.appGlue.DesignPaterns.factory import StaticFactory
from SLM.appGlue.core import Resource

class GlueObjectFieldDataScheme:
    def __init__(self):
        self.name = ""
        self.field_type: Optional[type] = None
        self.default = None

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def GetValue(self, obj_instance):
        if self.field_type == str:
            return getattr(obj_instance, self.name, self.default)

    def SetValue(self, obj_instance, value):
        if self.field_type == str:
            setattr(obj_instance, self.name, value)

class GlueObjectDataScheme:
    def __init__(self):
        self.name = ""
        self.dataclass = None
        self.fields = []

    @staticmethod
    def scheme_from_cls(cls, *args, **kwargs):
        scheme = GlueObjectDataScheme()
        scheme.name = cls.__name__
        scheme.dataclass = cls
        instance = cls(*args, **kwargs)
        # get list of public fields of instance of cls
        fields = [field for field in dir(instance) if
                  not field.startswith('_') and not callable(getattr(instance, field))]
        for field in fields:
            field_scheme = GlueObjectFieldDataScheme()
            field_scheme.name = field
            field_scheme.field_type = type(getattr(instance, field))
            field_scheme.default = getattr(instance, field)
            scheme.fields.append(field_scheme)
        return scheme

class DataAdapter:
    def __init__(self):
        self.converters = []

    def Convert(self, data_item, target_type):
        for converter in self.converters:
            if converter.sourcetpe == type(data_item) and converter.targettype == target_type:
                return converter.Convert(data_item, )
            if converter.targettype == type(data_item) and converter.sourcetpe == target_type:
                return converter.ConvertBack(data_item)
        return None

class DataSchemeMapper:
    def __init__(self):
        self.schemas = {}

    def map(self, source_instance, target_instance):
        if type(source_instance) in self.schemas and type(target_instance) in self.schemas:
            source_schema = self.schemas[type(source_instance)]
            target_schema = self.schemas[type(target_instance)]
            for field in source_schema.fields:
                if field.name in target_schema.fields:
                    target_field = target_schema.fields[field.name]
                    target_field.SetValue(target_instance, field.GetValue(source_instance))

    def AddSchema(self, schema):
        self.schemas[schema.dataclass] = schema

class DataProvider:
    def __init__(self):
        self.name = "DataProviderBase"
        self.dataformath = ''
        self.dataAdapter = DataAdapter()

        self.backStores: list[DataProvider] = []

    def GetDataItem(self, **kwargs):
        # abstract method
        return None

    def GetDataItemFormat(self, dataitem):
        pass

    def GetDataItemByBackStore(self, storename, **kwargs):
        for store in self.backStores:
            if store.name == storename:
                dataitem = store.GetDataItem(**kwargs)
                formath = store.GetDataItemFormat(dataitem)
                return self.dataAdapter.Convert(dataitem, formath)

    def SetDataItem(self, item, **kwargs):
        # abstract metod
        for store in self.backStores:
            store.SetDataItem(item, **kwargs)

    def deleteDataItem(self, item):
        # abstract metod
        for store in self.backStores:
            store.deleteDataItem(item)

    def all(sel, **kwargs):
        return None

    def OnDataChange(self, **kwargs):
        for store in self.backStores:
            store.OnDataChange(**kwargs)

    def InvokeDataChange(self, **kwargs):
        for store in self.backStores:
            store.InvokeDataChange(**kwargs)

    def commit(self):
        for store in self.backStores:
            store.commit()

class GlueDataConverter:

    def Convert(self, data):
        return None

    def ConvertBack(self, data):
        return None

class ForwardConverter(GlueDataConverter):
    def Convert(self, data):
        return data

    def ConvertBack(self, data):
        return data

class damyTostrConverter(ForwardConverter):
    def __init__(self):
        self.val_str = {}

    def Convert(self, data):
        str_val = str(data)
        self.val_str[str_val] = data
        return str_val

    def ConvertBack(self, data):
        return self.val_str[data]

class DataConverterFactory(StaticFactory, Resource):
    converter_dict = {}

    def __init__(self):
        super().__init__()
        self.name = "DataConverterFactory"

    def build(self, **kwargs):
        converter_name = kwargs.get("converter_name")
        if converter_name is None:
            return None
        return self.converter_dict[converter_name]

class AdapterTemplateSelector:
    """AdapterTemplateSelector is a class that helps in selecting a template based on the type of the item or a custom selector function.
    Attributes:
        template_map (dict[type, type]): A dictionary mapping item types to their corresponding templates.
        get_template_del (list[Callable[[object], object]]): A list of custom selector functions.
        default_template (type): The default template to use if no match is found.
    Methods:
        __init__(def_template=None):
            Initializes the AdapterTemplateSelector with an optional default template.
        add_template(ittype: type, template: type):
            Adds a template to the template_map for a specific item type.
        add_template_selector(selector: Callable[[object], object]):
            Adds a custom selector function to the get_template_del list.
        get_template(item) -> type:
            Returns the appropriate template for the given item based on the custom selectors and template_map."""

    def __init__(self, def_template=None):
        self.template_map: dict[type, type] = {}
        self.get_template_del = []
        if def_template is None:
            self.default_template = None
        else:
            self.default_template = def_template

    def add_template(self, ittype: type, template: type):
        self.template_map[ittype] = template

    def add_template_selector(self, selector: Callable[[object], object]):
        """
        Adds a custom selector function to the get_template_del list.

        Use for custom template selection.

        Args:
            selector (Callable[[object], object]): A function that takes an item and returns a template.

        Example:
            def selector(item):
                if isinstance(item, str):
                    return MyTemplate
                return None
        """
        self.get_template_del.append(selector)

    def get_template(self, item) -> Optional[type]:
        for delegate in self.get_template_del:
            template = delegate(item)
            if template is not None:
                return template
        for item_type in self.template_map.keys():
            if isinstance(item, item_type):
                return self.template_map[item_type]
        return self.default_template
