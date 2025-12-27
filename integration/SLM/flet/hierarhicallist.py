from typing import Any, Type

import flet as ft
from flet_core import ScrollMode

from SLM.appGlue.DesignPaterns.specification import AllSpecification
from SLM.flet.flet_ext import ftUserControl, ftUserRow


class TreeNode:
    def __init__(self, parent_tree=None, parent_node=None):
        self.name = ""
        self.update_suspend = False
        self.sub_nodes = []
        self.parent_node:TreeNode = parent_node
        self.parent_tree = parent_tree

    def append(self, node):
        self.get_sub_items().append(node)
        self.parent_tree.fire_tree_change("append", self, node)

    def remove(self, node):
        subitems = self.get_sub_items()
        if node in subitems:
            subitems.remove(node)
        self.parent_tree.fire_tree_change("remove", self, node)

    def extend(self, node_list):
        self.get_sub_items().extend(node_list)
        self.parent_tree.fire_tree_change("extend", self, node_list)

    def clear(self):
        self.get_sub_items().clear()
        self.parent_tree.fire_tree_change("clear", self, None)

    def get_sub_items(self):
        return self.sub_nodes

    def get_filtered_items(self):
        ret_list = []
        for node in self.get_sub_items():
            if self.parent_tree.specification.is_satisfied_by(node):
                ret_list.append(node)
        return ret_list


class ftTreeViewNode(ftUserRow, TreeNode):
    def __init__(self, data_item=None, parent_tree=None):
        super().__init__()
        self.exp_button = None
        self.count_text = ft.Text("")
        self.data = data_item
        self.parent_tree:TreeData = parent_tree
        self.parent_tree_view:ftTreeView = None
        self.expand_loose = True
        self.expand = False
        self.item_expand = False
        self.root_layout = ft.Column(spacing=0)
        self.controls.append(self.root_layout)
        self._header_layout = ft.Row(spacing=1)
        self._subitems_row = ft.Row(spacing=1)
        self._subitems_row.controls.append(ft.Column(width=20))
        self._sub_items_container = ft.Column(spacing=1) # important control space in vertical direction
        self._subitems_row.controls.append(self._sub_items_container)

        self.create_layout()
        self.build_node(data_item)

    def create_layout(self):
        self.root_layout.controls.append(self._header_layout)
        self.root_layout.controls.append(self._subitems_row)

    def refresh(self):
        if self.item_expand:
            self.clear()
            self.expand_sub_items()

    def soft_refresh_(self):
        if self.item_expand:
            # Convert current data items and nodes to sets for faster operations
            current_data_items = set(self.get_filtered_items())
            current_nodes = set(self._sub_items_container.controls)

            # Find deleted nodes using set difference
            deleted_nodes = current_nodes - current_data_items
            for node in deleted_nodes:
                self._sub_items_container.controls.remove(node)

            # Find added items and add them
            added_items = current_data_items - current_nodes
            for data_item in added_items:
                node = self.parent_tree_view.build_node(data_item)
                node.parent_node = self
                node.parent_tree = self.parent_tree
                node.parent_tree_view = self.parent_tree_view
                self._sub_items_container.controls.append(node)

            # Refresh updated nodes
            updated_nodes = self._sub_items_container.controls
            for node in updated_nodes:
                node.soft_refresh()

            # Update the UI once after all changes have been made
            self.update()

    def soft_refresh(self):
        # todo handle changing of items count and heandle not expanded for add expand button
        if self.item_expand:
            current_data_items = self.get_filtered_items()
            current_nodes = self._sub_items_container.controls
            for node in current_nodes: # data items deleted
                if node.data not in current_data_items:
                    self._sub_items_container.controls.remove(node)
            for data_item in current_data_items: # data items added
                if data_item not in [node.data for node in current_nodes]:
                    node = self.parent_tree_view.build_node(data_item)
                    node.parent_node = self
                    node.parent_tree = self.parent_tree
                    node.parent_tree_view = self.parent_tree_view
                    self._sub_items_container.controls.append(node)
            updated_nodes = self._sub_items_container.controls
            for node in updated_nodes:
                node.soft_refresh()
            self.update()


    def expand_click(self, event):
        if self.item_expand:
            self.collapse_sub_items()
        else:
            self.expand_sub_items()

    def expand_sub_items(self):
        self.exp_button.text = "-"
        progress_ring = ft.ProgressRing(value=None)
        self._header_layout.controls.append(progress_ring)
        self.update()

        for data_item in self.get_filtered_items():
            node: ftTreeViewNode = self.parent_tree_view.build_node(data_item)
            node.parent_node = self
            self.append(node)

        self.item_expand = True
        self._header_layout.controls.remove(progress_ring)
        self.update()

    def collapse_sub_items(self):
        self.exp_button.text = "+"
        self.clear()
        self.item_expand = False
        self.update()

    def set_header(self, header):
        self._header_layout.controls.clear()
        self._header_layout.controls.append(self.exp_button)
        self._header_layout.controls.append(header)
        self.update()

    def build_node(self, data_item):
        self.data = data_item
        self._header_layout.controls.clear()
        self.build_expand()
        self._header_layout.controls.append(self.build_header())


    def build_expand(self):
        sub_items = self.get_sub_items()
        if len(sub_items) != 0:
            self.exp_button = ft.TextButton("+",width=30)
            self.exp_button.on_click = self.expand_click
            self._header_layout.controls.append(self.exp_button)
            self.count_text = ft.Text(str(len(sub_items)))
            self._header_layout.controls.append(self.count_text)

    def build_header(self):
        return ft.Text(str(self.data))

    def get_sub_items(self):
        return []


class HierarchicalItemTemplateSelector:
    def __init__(self):
        self.template_map: dict[type] = {}
        self.default_template = ftTreeViewNode

    def add_template(self, ittype: type, template: ftTreeViewNode):
        self.template_map[ittype] = template

    def get_template(self, item) -> Type[ftTreeViewNode] | Any:
        for type in self.template_map.keys():
            if isinstance(item, type):
                return self.template_map[type]
        return self.default_template


class TreeViewTemplate:
    def __init__(self):
        self.list_widget = None
        self.itemTemplateSelector = HierarchicalItemTemplateSelector()

    def create_toolbar(self, list_widget):
        pass

    def create_items_layout(self, list_widget):
        pass


class TreeData(TreeNode):
    def __init__(self):
        super().__init__(self)
        self.observers = []
        self.specification = AllSpecification()

    def fire_tree_change(self, change_type, parent_node, child_node):
        for observer in self.observers:
            observer.tree_change(change_type, parent_node, child_node)


class ftTreeView(ftUserControl):

    def __init__(self):
        self.viewTemplate = TreeViewTemplate()
        self.groups_column = ft.Column(expand=True)
        super().__init__()
        self.expand_loose = True
        self.on_view_changed_suspend = False
        self.groups_column.scroll = ScrollMode.AUTO
        self.stack.controls.append(self.groups_column)
        self.treeData = TreeData()
        self.treeData.observers.append(self)
        self.item_node_dict = {}

    def build_node(self, data_item):
        node_type = self.viewTemplate.itemTemplateSelector.get_template(data_item)
        node = node_type(data_item)
        return node

    def add_item(self, data_item):
        node = self.build_node(data_item)
        self.treeData.append(node)

    def tree_change(self, change_type, parent_node, child_node):
        # todo posible need safe update as un list view
        if change_type == "append":
            self.item_node_dict[child_node.data] = child_node
            #??
            child_node.parent_tree = self.treeData
            child_node.parent_tree_view = self
            if isinstance(parent_node, TreeData):
                self.groups_column.controls.append(child_node)
                self.update()
            else:
                parent_node._sub_items_container.controls.append(child_node)
                parent_node.update()
        elif change_type == "clear":
            if isinstance(parent_node, TreeData):
                self.groups_column.controls.clear()
                self.update()
            else:
                parent_node._sub_items_container.controls.clear()
                parent_node.update()
        elif change_type == "remove":
            if isinstance(parent_node, TreeData):
                self.groups_column.controls.remove(child_node)
                self.update()
            else:
                child_node = self.item_node_dict[child_node]
                parent_node._sub_items_container.controls.remove(child_node)
                self.page.update()

    def soft_refresh(self):
        for node in self.groups_column.controls:
            node.soft_refresh()
        self.update()
