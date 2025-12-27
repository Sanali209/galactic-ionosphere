import subprocess

import pyperclip

from SLM.flet.flet_ext import flet_dialog_alert
from SLM.flet.hierarhicallist import ftTreeViewNode
from applications.collectionTools.views.collection_list_view.NavTree.items_models import FsRoot, FsFolderInfo
import flet as ft

from applications.collectionTools.views.collection_list_view.collectionView.listViewModel import EntityListViewModel
from applications.collectionTools.views.collection_list_view.tools.similar_items_explorer.similar_item_explorer_f import \
    SimilarItemsExplorerModel


class FsRootItemNode(ftTreeViewNode):
    def build_header(self):
        naw_button = ft.TextButton(self.data.name)
        naw_button.on_click = self.on_click
        return naw_button

    def on_click(self, event):
        EntityListViewModel().set_folder_query(None)

    def get_sub_items(self):
        data_item: FsRoot = self.data
        folders = data_item.get_subFolders_info()
        return folders


class RecordTypeRootItemNode(ftTreeViewNode):
    def build_header(self):
        return ft.Text(self.data.name)

    def get_sub_items(self):
        return list(self.data.record_types)


class RecordTypeModelItemNode(ftTreeViewNode):
    def build_header(self):
        button = ft.TextButton(self.data.name)
        button.on_click = self.on_click

        return button

    def on_click(self, event):
        EntityListViewModel().set_record_type_query(self.data.name)


class FsFolderInfoItemNode(ftTreeViewNode):
    def build_header(self):
        row = ft.Row(spacing=1)
        naw_button = ft.TextButton(self.data.name)
        naw_button.on_click = self.on_click
        row.controls.append(naw_button)
        pop_up_menu = ft.PopupMenuButton(
            items=[ft.PopupMenuItem(text="Show in explorer", on_click=self.on_show_explorer),
                   ft.PopupMenuItem(text="Copy path to clipboard",
                                    on_click=self.on_copy_path),
                   ft.PopupMenuItem(text="Show recusive", on_click=self.on_show_recursive),
                   ft.PopupMenuItem(text="set on similar items explorer", on_click=self.on_set_similar_items_explorer)
                   ])
        row.controls.append(pop_up_menu)
        return row

    def on_set_similar_items_explorer(self, event):
        path = self.data.path
        path = path.strip("/")
        path = path.replace("/", "\\")
        SimilarItemsExplorerModel().set_path(path)

    def get_sub_items(self):
        data_item: FsFolderInfo = self.data
        folders = data_item.get_subFolders_info()
        return folders

    def on_click(self, event):
        path = self.data.path
        path = path.strip("/")
        path = path.replace("/", "\\")
        EntityListViewModel().set_folder_query(path)

    def on_show_recursive(self, event):
        path = self.data.path
        path = path.strip("/")
        path = path.replace("/", "\\\\")
        EntityListViewModel().set_folder_query(path, True)

    def on_copy_path(self, event):
        path = self.data.path
        path = path.strip("/")
        path = path.replace("/", "\\\\")
        pyperclip.copy(path)

    def on_show_explorer(self, event):
        # info need convert linux form path convert to windows format
        path = self.data.path
        path = path.replace("/", "\\")
        if len(path) > 3:
            path = path.strip("\\")
        subprocess.run(['explorer', path])
