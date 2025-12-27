from typing import Callable

import flet as ft

from SLM.appGlue.DAL.DAL import AdapterTemplateSelector
from SLM.flet.flet_ext import ftUserControl, ftUserColumn


class ObjectEditorView(ftUserColumn):
    def __init__(self, edit_object: object = None):
        super().__init__()
        self.object: object = edit_object
        self.expand = True
        self.scroll = ft.ScrollMode.AUTO




class FtObjectEditorTemplate:
    '''Template for the object editor control behavior of the FtObjectEditor control'''
    def __init__(self):
        self.list_widget = None
        self.item_view_template = AdapterTemplateSelector()

    def create_toolbar(self, list_widget):
        pass



class FtObjectEditor(ftUserColumn):

    def __init__(self, **kwargs):
        super().__init__()
        self.control_template = FtObjectEditorTemplate()
        self.expand = True  # flet layout expand
        self._enable = True  # if the control is enabled
        self.sel_object = None

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = value
        if not self._enable:
            self.clear()

    def add_view_template(self, selector: Callable[[object], object]):
        self.control_template.item_view_template.add_template_selector(selector)

    def set_object(self, obj):
        self.sel_object = obj
        view = self.control_template.item_view_template.get_template(obj)
        if view is None:
            return None
        else:
            view = view(obj)
        self.controls.clear()
        self.controls.append(view)
        view.object = obj
        self.update()

    def switch_editor_template(self, template: FtObjectEditorTemplate):
        self.control_template = template
        self.controls.clear()
        self.set_object(self.sel_object)


    def clear(self):
        """Clears the controls and updates the view.
        Update with flet update"""
        self.controls.clear()
        self.update()


