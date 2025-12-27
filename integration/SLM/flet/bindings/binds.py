from SLM.appGlue.DAL.DAL import ForwardConverter, damyTostrConverter
from SLM.appGlue.DAL.binding.bind import BindingObserver, BindFactoryBase
import flet as ft
"""
sample of binding for flet

create bind class
class group_editor_bindings(PropUser):
    current_job: AnnotationJob = PropInfo()

    def __init__(self):
        super().__init__()
        self.current_job = all_jobs[0]
        
create binding user class
class SingleAnnotationView(Flet_view):
    active_job_dropdown: ft.Dropdown

    def __init__(self):
        super().__init__()
        self.prop = group_editor_bindings()
        self.xml_source = ""
        <root>
            <DropDown id="active_job_dropdown" value="bind:current_job" />
        </root>
        ""
        self.parse_xml()
        
        create binding
        self.prop.dispatcher.current_job.bind(bind_target=self.active_job_dropdown, field="value")

"""

class ftTextFieldBind(BindingObserver):
    """
    binding for ft.TextField
    params:
    bind_target: ft.TextField
    todo: add converter
    """
    def __init__(self, bind_target: ft.TextField = None, *args, **kwargs):
        super().__init__()
        self.ft_text_field = bind_target
        if self.ft_text_field is None:
            self.ft_text_field = kwargs.get("bind_target")
        self.ft_text_field.on_change = self.on_text_change

    def on_registered(self):
        self.ft_text_field.value = self.get_prop_val()

    def on_text_change(self, e):
        self.set_prop_val(e.control.value)

    def on_prop_changed(self, prop, val):
        self.ft_text_field.value = val
        self.ft_text_field.update()


BindFactoryBase.bind_dict[ft.TextField] = ftTextFieldBind


class ftTextBind(BindingObserver):
    """
    binding for ft.Text
    params:
    bind_target: ft.Text
    todo: add converter

    """
    def __init__(self, bind_target: ft.Text = None, *args, **kwargs):
        super().__init__()
        self.ft_text = bind_target
        if self.ft_text is None:
            self.ft_text = kwargs.get("bind_target")

    def on_registered(self):
        self.ft_text.value = self.get_prop_val()

    def on_prop_changed(self, prop, val):
        self.ft_text.value = val
        self.ft_text.update()


BindFactoryBase.bind_dict[ft.Text] = ftTextBind


class BaseIterableToStrConverter(ForwardConverter):
    def __init__(self, options):
        self.val_str = {}
        for i, val in enumerate(options):
            self.val_str[str(val)] = val

    def Convert(self, val):
        return str(val)

    def ConvertBack(self, val):
        return self.val_str[val]


class FtDropdownBind(BindingObserver):
    """
    binding for ft.Dropdown
    params:
    bind_target: ft.Dropdown
    converter: Converter
    callback: callable
    field: str - "value" or "options"
    """

    def __init__(self, bind_target: ft.Dropdown, converter=None, callback=None, field="value"):
        super().__init__()
        self.ft_dropdown = bind_target
        self.field = field
        self.callback = callback
        self.converter = converter
        if field == "value":
            self.ft_dropdown.on_change = self.on_change
            if self.converter is None:
                self.converter = ForwardConverter()
        elif field == "options":
            if self.converter is None:
                self.converter = damyTostrConverter()

    def on_registered(self):
        if self.field == "value":
            self.ft_dropdown.value = self.get_prop_val()
        if self.field == "options":
            self.ft_dropdown.options = self.get_prop_val()
        super().on_registered()

    def on_change(self, e):
        self.set_prop_val(e.control.value)

    def on_prop_changed(self, prop, in_val):
        if self.field == "value":
            self.ft_dropdown.value = in_val
        if self.field == "options":
            self.ft_dropdown.options = in_val
        self.ft_dropdown.update()


BindFactoryBase.bind_dict[ft.Dropdown] = FtDropdownBind
