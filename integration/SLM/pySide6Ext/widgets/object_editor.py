from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PySide6.QtCore import Qt

from SLM.appGlue.DAL.DAL import AdapterTemplateSelector
from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget

class ObjectEditorView(PySide6GlueWidget):
    """Gui for editing an object - defain on construction"""
    def __init__(self, edit_object: object = None):
        super().__init__()
        self.object: object = edit_object



class PySide6ObjectEditorTemplate:
    def __init__(self):
        self.item_view_template:AdapterTemplateSelector = AdapterTemplateSelector()


class PySide6ObjectEditor(PySide6GlueWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.control_template:PySide6ObjectEditorTemplate = PySide6ObjectEditorTemplate()
        self.sel_object = None
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)


    def add_view_template(self, item_type, view_template):
        """Registers a template creator based on a selector function."""
        self.control_template.item_view_template.add_template(item_type, view_template)

    def set_object(self, obj):
        """Sets the object to be edited and updates the view."""
        self.sel_object = obj
        view = self.control_template.item_view_template.get_template(obj)
        if view is None:
            return None
        else:
            self.clear()
            template_widget = view(obj)

            self.main_layout.addWidget(template_widget)
            #template_widget.setParent(self)

    def switch_editor_template(self, template):
        """Switches to a different editor template."""
        self.control_template = template
        self.set_object(self.sel_object)

    def clear(self):
        """Clears the editor view."""
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
