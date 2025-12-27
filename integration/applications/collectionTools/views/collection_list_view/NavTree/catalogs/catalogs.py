from SLM.files_db.components.catalogs.catalog import CatalogRecord
from SLM.files_db.components.fs_tag import TagRecord
from SLM.flet.flet_ext import flet_dialog_alert
from SLM.flet.hierarhicallist import ftTreeViewNode
import flet as ft

from applications.collectionTools.views.collection_list_view.collectionView.listViewModel import EntityListViewModel
from applications.collectionTools.views.collection_list_view.selectionManager.selection_manager import SelectionManager




class CatalogRoot:
    def __init__(self):
        self.name = "Catalog"

    def get_subFolders_info(self):
        all_tags = CatalogRecord.get_all_catalogs(True)
        return all_tags


class CatalogRootItemNode(ftTreeViewNode):
    def build_header(self):
        row = ft.Row()
        button = ft.TextButton(self.data.name)
        button.on_click = self.on_click
        row.controls.append(button)
        pop_up_menu = ft.PopupMenuButton(items=[ft.PopupMenuItem(text="Add", on_click=self.on_add),
                                                ft.PopupMenuItem(text="Delete",
                                                                 on_click=lambda event: print("delete"))])
        row.controls.append(pop_up_menu)
        return row

    def get_sub_items(self):
        return self.data.get_subFolders_info()

    def on_click(self, event):
        EntityListViewModel().clear_tag_query()

    def on_add(self, event):
        dialog = flet_dialog_alert("enter path", "")
        text_input = ft.TextField()

        def on_b_click(event):
            TagRecord.get_or_create(full_name=text_input.value)
            self.refresh()

        dialog.set_content(ft.Column([
            ft.Text("Enter tsg name"),
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()


class TagItemNode(ftTreeViewNode):
    def build_header(self):
        self.hashed_tags = []
        row = ft.Row(spacing=1)
        naw_button = ft.TextButton(self.data.name)
        row.controls.append(naw_button)
        naw_button.on_click = self.on_click
        tagIcon = ft.IconButton(icon=ft.icons.BOOKMARK_ADD, on_click=self.on_tag_selected)
        untagIcon = ft.IconButton(icon=ft.icons.BOOKMARK_REMOVE, on_click=self.on_un_tag_selected)
        row.controls.extend([tagIcon, untagIcon])

        pop_up_menu = ft.PopupMenuButton(items=[ft.PopupMenuItem(text="Add", on_click=self.on_add),
                                                ft.PopupMenuItem(text="Delete", on_click=self.on_delete),
                                                ft.PopupMenuItem(text="Rename", on_click=self.on_rename),
                                                ])
        row.controls.append(pop_up_menu)
        return row

    def get_sub_items(self):
        data_item: TagRecord = self.data
        res = data_item.find({"parent_tag": data_item._id}, {"fullName": 1})
        return res

    def on_click(self, event):
        full_name = self.data.fullName
        EntityListViewModel().set_tag_query(full_name)

    def on_add(self, event):
        # todo improve
        dialog = flet_dialog_alert("enter path", "")
        text_input = ft.TextField(value=self.data.fullName)

        def on_b_click(event):
            TagRecord.get_or_create(full_name=text_input.value)
            self.parent_tree_view.soft_refresh()
            dialog.close_dlg()

        dialog.set_content(ft.Column([
            ft.Text("Enter tsg name"),
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()

    def on_rename(self, event):
        dialog = flet_dialog_alert("enter path", "")
        old_tag_name = self.data.fullName
        text_input = ft.TextField(value=self.data.fullName)

        def on_b_click(event):
            if old_tag_name == text_input.value:
                dialog.close_dlg()
                return
            tag = TagRecord.get_or_create(fullName=old_tag_name)
            tag.rename(text_input.value)
            self.parent_tree_view.soft_refresh()
            dialog.close_dlg()

        dialog.set_content(ft.Column([
            ft.Text("Enter tsg name"),
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()

    def on_delete(self, event):
        current_tag: TagRecord = self.data
        self.parent_node.remove(self.data)
        self.parent_tree_view.soft_refresh()
        current_tag.delete()

    def on_tag_selected(self, event):
        current_tag: TagRecord = self.data
        sel = SelectionManager().get_selection()
        for collection_record in sel:
            current_tag.add_to_file_rec(collection_record)

    def on_un_tag_selected(self, event):
        current_tag: TagRecord = self.data
        sel = SelectionManager().get_selection()
        for collection_record in sel:
            current_tag.remove_from_file_rec(collection_record)
