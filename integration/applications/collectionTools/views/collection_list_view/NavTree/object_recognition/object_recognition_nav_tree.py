from SLM.files_db.object_recognition.object_recognition import DetectionObjectClass, Recognized_object
from SLM.flet.flet_ext import flet_dialog_alert
from SLM.flet.hierarhicallist import ftTreeViewNode
import flet as ft


class RecognizedRoot:
    def __init__(self):
        self.name = "Objects"

    def get_subitems(self):
        all_class = DetectionObjectClass.find({})
        return all_class


class RecognizedRootItemNode(ftTreeViewNode):
    def build_header(self):
        row = ft.Row()
        text = ft.Text("Objects")
        row.controls.append(text)
        pop_up_menu = ft.PopupMenuButton(items=[ft.PopupMenuItem(text="Add", on_click=self.on_add),
                                                ft.PopupMenuItem(text="Delete",
                                                                 on_click=lambda event: print("delete")),
                                                ft.PopupMenuItem(text="Rename")])
        row.controls.append(pop_up_menu)
        return row

    def get_sub_items(self):
        return self.data.get_subitems()

    def on_click(self, event):
        pass

    # todo implement clear tag query

    def on_add(self, event):
        dialog = flet_dialog_alert("enter name", "")
        text_input = ft.TextField()

        def on_b_click(event):
            DetectionObjectClass.get_or_create(name=text_input.value)
            self.parent_node.refresh()

        dialog.set_content(ft.Column([
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()


class RecognizedObjectClassItemNode(ftTreeViewNode):
    def build_header(self):
        row = ft.Row()
        naw_button = ft.TextButton(self.data.name)
        naw_button.on_click = self.on_click
        row.controls.append(naw_button)
        pop_up_menu = ft.PopupMenuButton(items=[ft.PopupMenuItem(text="Add", on_click=self.on_add),
                                                ft.PopupMenuItem(text="Delete",
                                                                 on_click=lambda event: print("delete")),
                                                ft.PopupMenuItem(text="Rename")])
        row.controls.append(pop_up_menu)
        return row

    def get_sub_items(self):
        list_ = Recognized_object.find({"obj_class_id": self.data._id})
        return list_

    def on_click(self, event):
        pass

    def on_add(self, event):
        dialog = flet_dialog_alert("enter name", "")
        text_input = ft.TextField()

        def on_b_click(event):
            obj = Recognized_object.get_or_create(name=text_input.value)
            obj.obj_class = self.data._id
            self.parent_node.refresh()

        dialog.set_content(ft.Column([
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()


class RecognizedObjectItemNode(ftTreeViewNode):
    def build_header(self):
        row = ft.Row()
        naw_button = ft.TextButton(self.data.name)
        naw_button.on_click = self.on_click
        row.controls.append(naw_button)
        pop_up_menu = ft.PopupMenuButton(items=[ft.PopupMenuItem(text="Add selected", on_click=self.on_add_to_coll),
                                                ft.PopupMenuItem(text="Delete",
                                                                 on_click=lambda event: print("delete")),
                                                ft.PopupMenuItem(text="Rename")])
        row.controls.append(pop_up_menu)
        return row

    def on_add_to_coll(self, event):
        pass

    def get_sub_items(self):
        return []

    def on_delete(self, event):
        pass

    def on_click(self, event):
        pass
