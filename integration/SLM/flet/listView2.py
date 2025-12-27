import time
import uuid

import flet as ft
from loguru import logger

from SLM.appGlue.DAL.DAL import AdapterTemplateSelector
from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo
from SLM.appGlue.DAL.datalist2 import DataListModel, DataViewCursor
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.timertreaded import Timer
from SLM.destr_worck.bg_worcker import BGTask, BGWorker, TaskQueueSeq
from SLM.flet.flet_ext import ftUserControl, ftUserColumn


# todo list of sort orders

class ListViewItemWidget(ftUserColumn):

    def __init__(self, **kwargs):
        super().__init__()
        self.alignment = ft.alignment.center
        self.width = 256
        self.expand = False # overide?
        self.parent_list_view:ftListWidget = kwargs.get("list_widget")
        cont = ft.Container(alignment=ft.alignment.center,expand=False)
        #row = ft.Column(expand_loose=False, wrap=True, spacing=1, run_spacing=1)
        self.main_container = ft.Card(
            color=ft.colors.ON_PRIMARY
        )
        cont.on_click = self.on_click
        cont.on_long_press = self.on_long_press
        self.expand_button = ft.TextButton('>',width=20,on_click=self.expand_refs)
        #row.controls.append(self.main_container)
        #row.controls.append(self.expand_button)
        cont.content = self.main_container
        self.controls.append(cont)
        self.data_context = None
        self.sub_items_show = False
        self.sub_items_layout = ft.Row(expand_loose=True, wrap=True, spacing=1, run_spacing=1)
        self.controls.append(self.sub_items_layout)
        self.sub = False

    def on_click(self, event):
        self.parent_list_view.select(self)
        self.parent_list_view.clicks_counter += 1
        self.parent_list_view.clicks_count(self)

    def on_clicks(self, clicks_count):
        pass

    def on_long_press(self, event):
        pass

    def build_widget(self, data_context):
        self.data_context = data_context
        self.build_header()

    def build_header(self):
        self.main_container.content = ft.Text(str(self.data_context))

    def get_group(self, group_param):
        return "defaulth"

    def get_refs(self):
        return []

    def expand_refs(self):
        for item in self.get_refs():
            template = self.parent_list_view.template.itemTemplateSelector.get_template(item)
            lv_item = template(list_widget=self)
            self.sub_items_layout.controls.append(lv_item)
        self.sub_items_show = True
        self.update()

    def collapse_refs(self):
        self.sub_items_layout.controls.clear()
        self.sub_items_show = False
        self.update()


class ListViewGroupWidget(ftUserColumn):
    def __init__(self, data_context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacing = 1
        self.expand_loose = True
        self.data_context = data_context
        self.build_header()
        self.items_layout = ft.Row(expand_loose=True, wrap=True, spacing=1, run_spacing=1)
        self.controls.extend([self.header, self.items_layout])

    def build_header(self):
        self.header = ft.Card(color=ft.colors.RED, content=ft.Text(str(self.data_context)))


class ListViewTemplate:
    def __init__(self):
        self.list_widget = None
        self.itemTemplateSelector = AdapterTemplateSelector(ListViewItemWidget)
        self.groupTemplateSelector = AdapterTemplateSelector(ListViewGroupWidget)

    def create_toolbar(self, list_widget):
        pass

    def create_items_layout(self, list_widget):
        pass


class ListWidgetProps(PropUser):
    current_page: int = PropInfo()
    page_count: int = PropInfo()


class ftListWidget(ftUserControl):

    def __init__(self, template: ListViewTemplate = None):
        super().__init__()
        self.prop = ListWidgetProps()
        self.template = template
        if self.template is None:
            self.template = ListViewTemplate()
        self.main_layout = ft.Column(expand=True, spacing=0)
        self.stack.controls.append(self.main_layout)
        self.expand_loose = True
        self.toolbar = ft.Row()
        self.items_layout = ft.Row(expand=True, wrap=True, spacing=0)
        self.items_layout.scroll = ft.ScrollMode.AUTO
        self.main_layout.controls.append(self.toolbar)
        self.main_layout.controls.append(self.items_layout)
        self.data_list: DataListModel = DataListModel()
        self.data_list_cursor = DataViewCursor(self.data_list)
        self.data_list_cursor.attach(self)
        self.items_groups = {}
        self.items_nodes = {}
        self.grouping_mode = None
        # prevent condition racing
        self.uuid = uuid.uuid4()
        BGWorker.instance().add_queue(self.uuid, TaskQueueSeq)
        self.selected_items = []
        self.selection_mode = "single"
        self.sel_message = "selection_changed"
        # double click heandler
        self.clicks_counter = 0
        self.clicks_count_timer = Timer(0.2)
        self.clicks_count_timer.single = True
        self.clicks_count_timer.register(self)
        self.list_changed_callbacks = []
        self.haiden_items = []

        self._call_on_selected = []

    def set_listener_on_selected(self, callback, delete=False):
        """
        Add callback to selection changed event
        :param callback: handler of selection changed event
        :param delete: if True remove callback
        :return:
        """
        if delete:
            self._call_on_selected.remove(callback)
        else:
            self._call_on_selected.append(callback)

    def _fire_on_selected(self, selection):
        for callback in self._call_on_selected:
            callback(selection)

    def clicks_count(self, list_item):
        self.clicks_count_timer.start()

    def on_timer_notify(self, timer):
        item_node: ListViewItemWidget = timer.owner
        item_node.on_clicks(self.clicks_counter)
        self.clicks_counter = 0

    def select(self, item):
        self.unhailait_selected()
        selected_data = []
        if self.selection_mode == "single":

            self.selected_items.clear()
            self.selected_items.append(item)
            selected_data = [x.data_context for x in self.selected_items]
        elif self.selection_mode == "multi":
            if item in self.selected_items:
                self.selected_items.remove(item)
            else:
                self.selected_items.append(item)
            selected_data = [x.data_context for x in self.selected_items]
        self.hailait_selected()
        self._fire_on_selected(selected_data)


    def get_selected(self):
        return [x.data_context for x in self.selected_items]

    def hailait_selected(self):
        for item in self.selected_items:
            item.main_container.color = ft.colors.PRIMARY_CONTAINER
            if item.main_container.page != None:
                item.main_container.update()

    def unhailait_selected(self):
        for item in self.selected_items:
            item.main_container.color = ft.colors.ON_PRIMARY
            if item.main_container.page != None:
                item.main_container.update()

    def select_all(self):
        self.selected_items.clear()
        sel_data = self.data_list_cursor.get_filtered_data(all_pages=True)
        for item in self.items_nodes.values():
            self.selected_items.append(item)
        self.hailait_selected()
        MessageSystem.SendMessage(self.sel_message, selection=sel_data)
        self._fire_on_selected(sel_data)
        self.update()

    def select_all_in_view(self):
        self.selected_items.clear()
        for item in self.items_nodes.values():
            self.selected_items.append(item)
        selected_data = [x.data_context for x in self.selected_items]
        MessageSystem.SendMessage(self.sel_message, selection=selected_data)
        self.hailait_selected()
        self.update()

    def select_clear(self):
        self.unhailait_selected()
        self.selected_items.clear()
        self.update()

    def get_current_page_data_items(self) -> list:
        ret_list = []
        for item in self.items_nodes.values():
            ret_list.append(item.data_context)
        return ret_list

    def set_data_list(self, data_list):
        self.data_list = data_list
        converter = self.data_list_cursor.data_converter
        self.data_list_cursor = DataViewCursor(self.data_list)
        self.data_list_cursor.attach(self)
        self.data_list_cursor.data_converter = converter
        self.data_list_cursor.refresh()

    def fire_list_changed(self, event):
        for callback in self.list_changed_callbacks:
            callback(event)

    def list_update(self, data_model, change_type, item=None):

        if change_type == "refresh":
            # do this on background
            task = refresh_list(self, data_model)
            BGWorker.instance().add_task(task)
        elif change_type == "add":
            task = add_list(self, item)
            BGWorker.instance().add_task(task)
        elif change_type == "remove":
            task = remove_list(self, item)
            BGWorker.instance().add_task(task)
        elif change_type == "clear":
            self.clear()
        self.fire_list_changed(change_type)

    def soft_refresh(self):
        current_items = self.data_list_cursor.get_filtered_data()
        current_nodes = self.items_layout.controls
        for node in current_nodes:  # data items deleted
            if node.data_context not in current_items:
                pass

    def clear(self):
        self.items_layout.controls.clear()
        self.items_groups.clear()
        self.haiden_items.clear()
        self.items_nodes.clear()
        self.update()


    def construct_item(self, item):
        template = self.template.itemTemplateSelector.get_template(item)
        lv_item: ListViewItemWidget = template(list_widget=self)
        lv_item.data_context = item
        self.haiden_items.extend(lv_item.get_refs())
        self.haiden_items= list(set(self.haiden_items))
        if item in self.haiden_items:
            return
        lv_item.build_widget(item)
        item_group_inst = lv_item.get_group(self.grouping_mode)
        self.items_nodes[item] = lv_item
        # search group if exist
        group_widget = self.items_groups.get(item_group_inst, None)
        if group_widget is None:
            group_widget: ListViewGroupWidget = (self.template.groupTemplateSelector.
                                                 get_template(item_group_inst)(item_group_inst))
            group_widget.parent_list = self
            self.items_layout.controls.append(group_widget)
            self.items_groups[item_group_inst] = group_widget
        group_widget.items_layout.controls.append(lv_item)

    def construct_and_insert(self,item,insert_after_item):
        template = self.template.itemTemplateSelector.get_template(item)
        lv_item: ListViewItemWidget = template(list_widget=self)
        lv_item.data_context = item
        self.haiden_items.extend(lv_item.get_refs())
        self.haiden_items= list(set(self.haiden_items))
        if item in self.haiden_items:
            return
        lv_item.build_widget(item)
        item_group_inst = lv_item.get_group(self.grouping_mode)
        self.items_nodes[item] = lv_item
        # search group if exist
        group_widget = self.items_groups.get(item_group_inst, None)
        if group_widget is None:
            group_widget: ListViewGroupWidget = (self.template.groupTemplateSelector.
                                                 get_template(item_group_inst)(item_group_inst))
            group_widget.parent_list = self
            self.items_layout.controls.append(group_widget)
            self.items_groups[item_group_inst] = group_widget
        group_widget.items_layout.controls.insert(group_widget.items_layout.controls.index(insert_after_item)+1,lv_item)


class refresh_list(BGTask):
    def __init__(self, list_widget, data_model):
        super().__init__()
        self.grouping_mode = None
        self.name = "refresh_list"
        self.cancel_token = True
        self.list_widget: ftListWidget = list_widget
        self.data_model = data_model
        self.queue_name = self.list_widget.uuid

    def task_function(self, ):
        self.list_widget.clear()
        for item in self.data_model:
            self.list_widget.construct_item(item)
        self.list_widget.update()
        yield "done"


class add_list(BGTask):
    def __init__(self, list_widget, data_model):
        super().__init__()
        hash = data_model.__hash__()
        self.name = "list_add" + str(hash)
        self.exclude_names = ["list_add" + str(hash)]
        self.list_widget: ftListWidget = list_widget
        self.data_model = data_model
        self.queue_name = self.list_widget.uuid

    def task_function(self, ):
        if self.list_widget.items_nodes.get(self.data_model, None) is not None:
            yield "done"
        self.list_widget.construct_item(self.data_model)

        self.list_widget.update()
        time.sleep(0.05)
        yield "done"


class remove_list(BGTask):
    def __init__(self, list_widget, data_model):
        super().__init__()
        self.name = "list_remove"
        self.list_widget: ftListWidget = list_widget
        self.data_model = data_model
        self.queue_name = self.list_widget.uuid

    def task_function(self, ):
        if self.data_model not in self.list_widget.items_nodes:
            yield "done"
        node = self.list_widget.items_nodes.pop(self.data_model)

        item_group_inst = node.get_group(self.list_widget.grouping_mode)
        group_widget = self.list_widget.items_groups.get(item_group_inst, None)
        if group_widget is not None:
            group_widget.items_layout.controls.remove(node)

            try:
                list_items = self.list_widget.data_list_cursor.get_filtered_data()
                items_on_data_source = len(list_items)
                items_on_view = len(self.list_widget.items_nodes.values())
                if items_on_data_source > items_on_view:
                    last_item = list_items[-1]
                    if self.list_widget.items_nodes.get(self.data_model, None) is None:
                        self.list_widget.construct_item(last_item)
            except IndexError:
                logger.debug("No items for merge")
            self.list_widget.update()

        yield "done"



