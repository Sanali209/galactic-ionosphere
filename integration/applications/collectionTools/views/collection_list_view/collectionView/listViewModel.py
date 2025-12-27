import os

from SLM.appGlue.DAL.DAL import GlueDataConverter
from SLM.appGlue.DAL.datalist2 import MongoDataModel
from SLM.appGlue.DesignPaterns import allocator
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.files_db.components.CollectionRecordScheme import FileTypeRouter
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.collectionItem import CollectionRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.object_recognition.object_recognition import Detection

from SLM.appGlue.glue_app import GlueApp
from SLM.flet.flet_ext import flet_dialog_alert
from SLM.flet.listView2 import ftListWidget, ListViewItemWidget
import flet as ft

from SLM.mongoext.modular_str_to_query import MongoDBQueryParser
from applications.collectionTools.components.event_dispatcher import EventDispatcher


from applications.collectionTools.views.edit_obj_view import obj_editor_view


class fsTypeConverter(GlueDataConverter):
    def Convert(self, data):
        try:
            wraped = CollectionRecord.get_record_wrapper2(data["_id"], data["item_type"])
        except Exception as e:
            return data
        return wraped

    def ConvertBack(self, data: CollectionRecord):
        return data.get_record_data()


class MongoQueryBuilder:
    def __init__(self):
        self.query_blocks = {}

    def set_block(self, name, query):
        self.query_blocks[name] = query

    def build_query(self):
        comp_query = {"$and": []}
        list_query = []
        for key, value in self.query_blocks.items():
            if value is not None:
                list_query.append(value)
        comp_query["$and"] = list_query
        return comp_query


class MongoQueryBlock:
    def __init__(self):
        self.name = ""
        self.q_value = {}


# todo implement mongo filtering as class
class EntityListViewModel:
    def __init__(self):
        self.all_pages_count_text = None
        self.current_page_text = None
        self.all_items_count_text = None
        # todo implement
        self.query_text = None
        self.selection_count_text = None
        MessageSystem.Subscribe("on_collection_record_deleted", self, self.on_collection_record_deleted)
        self.query_builder = MongoQueryBuilder()
        self.list_view = ftListWidget()
        #self.list_view.on_clik = lambda event:setattr(DocumentContext(),"document",self)
        self.list_view.data_list_cursor.data_converter = fsTypeConverter()
        self.list_view.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        self.list_view.template.itemTemplateSelector.add_template(Detection, DetectionListViewItemTemplate)
        db_name = DBSettings.get_db_name()
        self.list_view.set_data_list(MongoDataModel("mongodb://localhost:27017",
                                                    db_name,
                                                    "collection_records"))
        self.text_query = ""
        self.list_view.list_changed_callbacks.append(self.on_list_changed)
        self.list_view.grouping_mode = "fs_path"
        self.list_view.set_listener_on_selected(self.on_selected)

    def on_selected(self, *args, **kwargs):
        kwargs.setdefault("source", self)
        kwargs.setdefault("mode",self.list_view.selection_mode)
        EventDispatcher().dispatch_event("on_select", *args, **kwargs)

    def multi_sel_mode(self, enabled):
        from applications.collectionTools.views.collection_list_view.tools.similar_items_explorer.similar_item_explorer_f import \
            SimilarItemsExplorerModel
        if enabled:
            self.list_view.selection_mode = "multi"
            SimilarItemsExplorerModel().list_view.selection_mode = "multi"
        else:
            self.list_view.selection_mode = "single"
            SimilarItemsExplorerModel().list_view.selection_mode = "single"

    def on_collection_record_deleted(self, event):
        self.list_view.data_list_cursor.refresh()

    def set_folder_query(self, folder=None, recursive=False):
        if folder is None:
            self.query_builder.set_block("directory", None)
        else:
            if recursive:
                self.query_builder.set_block("directory", {"local_path": {"$regex": f"^{folder}"}})
            else:
                self.query_builder.set_block("directory", {"local_path": folder})

        self.refresh_list()

    def set_tag_query(self, tag=None):
        if tag is None:
            self.query_builder.set_block("tag", None)
        else:
            self.query_builder.set_block("tag", {"tags": tag})
        self.refresh_list()

    def clear_tag_query(self):
        self.query_builder.set_block("tag", None)
        self.refresh_list()

    def set_text_query(self, text):

        self.text_query = text
        self.refresh_list()

    def set_record_type_query(self, record_type=None):
        if record_type is None:
            self.query_builder.set_block("item_type", None)
        else:
            self.query_builder.set_block("item_type", {"item_type": record_type})
        self.refresh_list()

    def refresh_list(self):
        parser = MongoDBQueryParser()
        query_string = self.text_query
        try:
            query = parser.parse_query(query_string)
            self.query_builder.set_block("text", query)
            # composite query
            comp_query = self.query_builder.build_query()
            self.query_text.value = str(comp_query)
            print(comp_query)
            EntityListViewModel().list_view.data_list_cursor.set_specification(comp_query)
        except Exception as e:
            EntityListViewModel().list_view.data_list_cursor.set_specification({"no": "query"})
            self.query_text.value += "bad query"
            print("bad query")

    def on_list_changed(self, event):
        self.list_update_metric()

    def list_update_metric(self):
        items_all_count = EntityListViewModel().list_view.data_list_cursor.all_items_count()
        self.all_items_count_text.value = f"{items_all_count}"
        self.current_page_text.value = f"{EntityListViewModel().list_view.data_list_cursor.current_page}"
        self.all_pages_count_text.value = f"{EntityListViewModel().list_view.data_list_cursor.max_page}"
        if self.list_view.page is not None:
            self.list_view.page.update()


class lwItemTemplate(ListViewItemWidget):

    def build_header(self):
        file_record = self.data_context
        name_text = ft.Text(file_record.name)
        row = ft.Row(wrap=True, spacing=2)
        file_type = ft.Text(file_record.file_type)
        poup_menu_button = ft.PopupMenuButton(
            items=[ft.PopupMenuItem(text="Edit attributes", on_click=lambda event: self.on_edit_click()),
                   ft.PopupMenuItem(text="Locate on explorer", on_click=self.locate_on_explorer),
                   ft.PopupMenuItem(text="Mark as image set", on_click=self.marck_as_imSet),]
        )
        row.controls.extend([file_type, poup_menu_button])
        image = ft.Image(src=file_record.get_thumb("medium"), width=200, height=200, fit=ft.ImageFit.SCALE_DOWN,
                         )
        self.main_container.content = ft.Column([name_text, row, image], expand_loose=True, spacing=2,
                                                alignment=ft.alignment.center,
                                                horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    def marck_as_imSet(self,event):
        selection = self.parent_list_view.selected_items
        for item in selection:
            if item.data_context != self.data_context:
                RelationRecord.set_relation(self.data_context,item.data_context, "image_set")

    def get_group(self, group_param):
        if group_param == "fs_path":
            return self.data_context.local_path

        return super().get_group(group_param)

    def get_refs(self):
        file_record: FileRecord = self.data_context
        rels = RelationRecord.get_outgoing_relations(file_record,"image_set")
        res = []
        for rel in rels:
            rec = CollectionRecord.get_record_wrapper(rel.to_id)
            res.append(rec)
        return res

    def on_edit_click(self):
        objeditor = obj_editor_view()
        app = allocator.Allocator.get_instance(GlueApp)
        app.set_view(objeditor)
        objeditor.edit_object(self.data_context)

    def on_clicks(self, clicks_count):
        if clicks_count == 2:
            dialog = flet_dialog_alert("image", "")
            image = ft.Image(src=self.data_context.full_path, fit=ft.ImageFit.SCALE_DOWN)

            def on_ok_click(*args, **kwargs):
                dialog.close_dlg()

            dialog.set_content(
                ft.Column([
                    image,
                    ft.ElevatedButton(text="Ok", on_click=on_ok_click)
                ], width=1000)
            )
            dialog.show()

    def locate_on_explorer(self,event):
        os.system(f"explorer /select,{self.data_context.full_path}")


class DetectionListViewItemTemplate(ListViewItemWidget):
    def build_header(self):
        detection: Detection = self.data_context

        row = ft.Row(wrap=True, spacing=2)
        class_text = ft.Text(detection.object_class)
        popup_menu_button = ft.PopupMenuButton(
            items=[ft.PopupMenuItem(text="mark as wrong", on_click=lambda event: self.on_mark_as_wrong), ]
        )
        row.controls.extend([class_text, popup_menu_button])
        image = ft.Image(src=detection.obj_image_path, width=200, height=200, fit=ft.ImageFit.SCALE_DOWN,
                         )
        self.main_container.content = ft.Column([row, image], expand_loose=True, spacing=2,
                                                alignment=ft.alignment.center,
                                                horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def on_mark_as_wrong(self, event):
        pass
