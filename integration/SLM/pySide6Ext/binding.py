from PySide6.QtWidgets import QLineEdit, QComboBox

from SLM.appGlue.DAL.DAL import ForwardConverter, damyTostrConverter
from SLM.appGlue.DAL.binding.bind import BindingObserver, BindFactoryBase


class QtTextFieldBind(BindingObserver):
    """
    binding for ft.TextField
    params:
    bind_target: ft.TextField
    todo: add converter
    """

    def __init__(self, bind_target: QLineEdit = None, *args, **kwargs):
        super().__init__()
        self.ft_text_field = bind_target
        if self.ft_text_field is None:
            self.ft_text_field = kwargs.get("bind_target")
        self.ft_text_field.textChanged.connect(self.on_text_change)

    def on_registered(self):
        self.ft_text_field.setText(self.get_prop_val())

    def on_text_change(self, e):
        self.set_prop_val(self.ft_text_field.text())

    def on_prop_changed(self, prop, val):
        self.ft_text_field.setText(val)


BindFactoryBase.bind_dict[QLineEdit] = QtTextFieldBind


class QTDropdownBind(BindingObserver):
    """
    binding for ft.Dropdown
    params:
    bind_target: ft.Dropdown
    converter: Converter
    callback: callable
    field: str - "value" or "options"
    """

    def __init__(self, bind_target: QComboBox, converter=None, callback=None, field="value"):
        super().__init__()
        self.ft_dropdown = bind_target
        self.field = field
        self.callback = callback
        self.converter = converter
        if field == "value":
            self.ft_dropdown.currentIndexChanged.connect(self.on_change)
            if self.converter is None:
                self.converter = ForwardConverter()
        elif field == "options":
            if self.converter is None:
                self.converter = damyTostrConverter()

    def on_registered(self):
        if self.field == "value":
            self.ft_dropdown.setCurrentText(self.get_prop_val())
        if self.field == "options":
            self.ft_dropdown.addItems(self.get_prop_val())
        super().on_registered()

    def on_change(self, e):
        text = self.ft_dropdown.currentText()
        self.set_prop_val(text)

    def on_prop_changed(self, prop, in_val):
        if self.field == "value":
            self.ft_dropdown.setCurrentText(in_val)
        if self.field == "options":
            self.ft_dropdown.clear()
            self.ft_dropdown.addItems(in_val)
        self.ft_dropdown.update()


BindFactoryBase.bind_dict[QComboBox] = QTDropdownBind


def binding_load():
    #TODO: implemetation is fake function for loading in import
    pass
