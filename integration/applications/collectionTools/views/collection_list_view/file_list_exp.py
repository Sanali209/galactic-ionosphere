import os

from SLM.appGlue.DesignPaterns import allocator
from SLM.files_db.components.File_record_wraper import FileRecord
#https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformers-for-image-classification#multi-label-image-classification

import flet as ft
from loguru import logger

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.appGlue.timertreaded import Timer

from SLM.appGlue.progress_visualize import ProgressManager
from SLM.destr_worck.bg_worcker import BGWorker

from SLM.flet.flet_ext import Flet_view, flet_dialog_alert, FletGlueApp, ftCollapsiblePanelVertical

from applications.collectionTools.views.collection_list_view.NavTree.navTree import NavTreeSystem
from applications.collectionTools.views.collection_list_view.collectionView.listViewModel import EntityListViewModel
from applications.collectionTools.views.collection_list_view.propEditPanel.prop_edit_panel import prop_editor_model
from applications.collectionTools.views.collection_list_view.selectionManager.selection_manager import SelectionManager
from applications.collectionTools.views.collection_list_view.tascks import add_folder_to_db_task, index_db_task
from applications.collectionTools.views.collection_list_view.tools.similar_items_explorer.similar_item_explorer_f import \
    SimilarItemsExplorerModel
from applications.collectionTools.views.edit_obj_view import obj_editor_view
from applications.collectionTools.views.half_auto_annotation.half_auto_anotation_flet import HalfAutoAnnotationView
from applications.collectionTools.views.single_anotation.single_anotation_flet import SingleAnnotationView


# todo context menu locate in explorer
# todo block user interface on annotation queue

class FileListEditorBindings(PropUser):
    query_string: str = PropInfo()

    def __init__(self):
        super().__init__()
        self.query_string = ""


class FileListEditorView(Flet_view):
    toolbar_row: ft.Row
    center_column: ft.Column
    gen_progress_message: ft.Text
    gen_view_progress_bar: ft.ProgressBar
    left_column: ft.Column
    right_column: ft.Column
    query_input_text_field: ft.TextField
    all_items_count_text: ft.Text
    current_page_text: ft.Text
    all_pages_count_text: ft.Text
    page_next_button: ft.ElevatedButton
    page_prev_button: ft.ElevatedButton
    dinamic_single_ann_menu: ft.SubmenuButton
    query_input_text: ft.Text

    def add_folder_to_db(self, event):
        dialog = flet_dialog_alert("enter path", "")
        text_inputh = ft.TextField()

        def on_b_click(event):
            dialog.close_dlg()

            path = text_inputh.value
            task = add_folder_to_db_task(path=path)
            BGWorker().add_task(task)

        dialog.set_content(ft.Column([
            ft.Text("Enter path"),
            text_inputh,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()

    def refresh_image_search(self,event):
        SimilarItemsExplorerModel().refreh_scope()

    def edit_db_file(self, event):
        dialog = flet_dialog_alert("enter path", "")
        text_input = ft.TextField()

        def on_b_click(event):
            obj_edit_v = obj_editor_view()
            self.app.set_view(obj_edit_v)
            file_obj = FileRecord.add_file_record_from_path(text_input.value)
            obj_edit_v.edit_object(file_obj)
            dialog.close_dlg()

        dialog.set_content(ft.Column([
            ft.Text("Enter path"),
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()

    def move_selected_to_folder(self, event):
        # todo refresh list
        selection = SelectionManager().selection
        # todo realise more acceptable dialog
        dialog = flet_dialog_alert("enter path", "")
        text_input = ft.TextField()

        def on_b_click(event):
            dialog.close_dlg()
            path = text_input.value
            # logic of movement in db
            for file in selection:
                try:
                    file.move_to_folder(path)
                except Exception as e:
                    logger.exception(e)

        dialog.set_content(ft.Column([
            ft.Text("Enter path"),
            text_input,
            ft.ElevatedButton(text="Ok", on_click=on_b_click)
        ], width=500))
        dialog.show()

    def __init__(self):
        super().__init__()
        self.prop = FileListEditorBindings()
        prog_man = ProgressManager.instance()
        prog_man.add_visualizer(self)
        self.expand = True
        # load source from xml

        with open("view_ui.xml", "r") as f:
            self.xml_source = f.read()
        self.parse_xml()
        self.sel_manager = SelectionManager()
        self.left_column.controls.append(NavTreeSystem().instance().naw_tree)

        self.tageditTab = ft.Tab(text="props")

        self.tageditTab.content = ft.Column(controls=[prop_editor_model().mode_select_control,
                                                      prop_editor_model().prop_editor])

        self.right_column.controls.append(ft.Tabs(expand=1, tabs=[self.tageditTab, ft.Tab(text="2")]))

        EntityListViewModel().list_view.expand = 3
        EntityListViewModel().all_pages_count_text = self.all_pages_count_text
        EntityListViewModel().current_page_text = self.current_page_text
        EntityListViewModel().all_items_count_text = self.all_items_count_text
        EntityListViewModel().query_text = self.query_input_text

        self.center_column.controls.append(EntityListViewModel().list_view)

        self.relation_panel = ftCollapsiblePanelVertical(label="Tools", expand=1)
        self.relation_panel.mode = "horizontal"

        self.relation_panel.set_content(SimilarItemsExplorerModel().list_view)
        self.center_column.controls.append(self.relation_panel)

        self.prop.dispatcher.query_string.add_listener(self.on_query_string_change)
        self.prop.dispatcher.query_string.bind(self.query_input_text_field)
        # prevent no need updates
        self.input_filter_timer = Timer(0.5)
        self.input_filter_timer.single = True
        self.input_filter_timer.register(self)
        self.buffered_string = ""
        EntityListViewModel().set_text_query(self.buffered_string)

    def open_single_annotation_tool(self, event):
        single_annotation_view = SingleAnnotationView()
        self.app.set_view(single_annotation_view)

    def open_half_auto_annotation_tool(self, event):
        toolView = HalfAutoAnnotationView()
        self.app.set_view(toolView)

    def list_mode_select(self, event):
        EntityListViewModel().multi_sel_mode(event.control.value)

    # region menu handlers
    def on_list_view_select_all(self, event):
        EntityListViewModel().list_view.select_all()

    def on_list_view_select_view(self, event):
        EntityListViewModel().list_view.select_all_in_view()

    def index_db(self, event):
        BGWorker().add_task(index_db_task())

    # endregion

    def on_page_next_button_click(self, event):
        EntityListViewModel().list_view.data_list_cursor.page_next()
        self.update()

    def on_page_prev_button_click(self, event):
        EntityListViewModel().list_view.data_list_cursor.page_previous()
        self.update()

    def on_query_string_change(self, event):
        self.buffered_string = self.prop.query_string
        self.input_filter_timer.stop()
        self.input_filter_timer.start()
        # file_type = image

    def on_timer_notify(self, timer):
        print("timer")
        # todo execute separated query
        EntityListViewModel().set_text_query(self.buffered_string)

    def update_progress(self):
        prog_man = ProgressManager.instance()
        cur_prog = prog_man.progress / prog_man.max_progress
        self.gen_progress_message.value = prog_man.message
        self.gen_view_progress_bar.value = cur_prog
        self.update()


if __name__ == "__main__":
    config = allocator.Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    config.documentConfig.path = r"D:\data\ImageDataManager"


    app = FletGlueApp()
    view = FileListEditorView()
    app.set_view(view)
    app.run()
    BGWorker.instance().stop()
