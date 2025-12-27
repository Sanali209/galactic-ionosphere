from typing import Union

import flet as ft

from SLM.appGlue.DAL.binding.bind import BindingProperty


class TypedDict(dict):
    list_view: ft.ListView


class ListViewItemWidget(ft.UserControl):

    def __init__(self, **kwargs):
        super().__init__()
        self.parent_list_view = kwargs.get("list_view")
        self.main_layout = ft.Column()
        self.header_layout = ft.Row()
        self.main_layout.controls.append(self.header_layout)

    def build(self):
        return self.main_layout


class ListViewItemBuilder:
    def build(self, list_view, data_item):
        lvitem = ListViewItemWidget(list_widget=list_view)
        lvitem.dataContext = data_item
        lvitem.navigationName = self.get_nav_name(data_item)
        self.build_control(lvitem, data_item)

        sub_items = self.get_sub_items(data_item)
        if sub_items is not None:
            lvitem.set_sub_items(sub_items)
        return lvitem

    def build_control(self, item_widget, data_item):
        item_widget.main_layout.height = 20
        button = ft.FilledButton(text=str(data_item))
        item_widget.header_layout.controls.append(button)

    def get_group_name(self, data_item, group_by) -> Union[None, list[str]]:
        return None

    def sort_by(self, data_item, sort_by):
        return data_item

    def get_sub_items(self, data_item) -> Union[None, list]:
        return None

    def get_nav_name(self, data_item):
        return ""


class ItemTemplateSelector:
    def __init__(self):
        self.template_map: dict[type] = {}
        self.get_template_del = []
        self.default_template = ListViewItemBuilder()

    def add_template(self, ittype: type, template: ListViewItemBuilder):
        self.template_map[ittype] = template

    def get_template(self, item) -> ListViewItemBuilder:
        for delgate in self.get_template_del:
            template = delgate(item)
            if template is not None:
                return template
        for type in self.template_map.keys():
            if isinstance(item, type):
                return self.template_map[type]
        return self.default_template


class ListViewTemplate:
    def __init__(self):
        self.list_widget = None
        self.itemTemplateSelector = ItemTemplateSelector()

    def create_toolbar(self, list_widget):
        pass

    def create_items_layout(self, list_widget):
        pass


class ListWidget(ft.UserControl):

    def __init__(self, template: ListViewTemplate = None):
        super().__init__()
        self.page: BindingProperty = BindingProperty()
        self.page_count: BindingProperty = BindingProperty()
        self.template = template
        if self.template is None:
            self.template = ListViewTemplate()
        self.main_layout = ft.Column(spacing=0)
        self.toolbar = ft.Row()
        self.items_layout = ft.GridView(expand=1, max_extent=150, height=200)
        self.main_layout.controls.append(self.toolbar)
        self.main_layout.controls.append(self.items_layout)

    def build(self):
        return self.main_layout

    def Refresh(self):
        self.Clear_layout()

    def SwitchTemplate(self, template: ListViewTemplate):
        self.template = template
        self.template.list_widget = self
        self.Refresh()

    def Clear_layout(self):
        if self.items_layout is not None:
            self.items_layout.clear_widgets()
        self.main_layout.clear_widgets()


if __name__ == "__main__":
    pass
