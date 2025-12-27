import re

from SLM.appGlue.DesignPaterns.SingletonAtr import singleton
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.vector_db_ext.vector_db_ext import SearchScopeMongoDb

from SLM.flet.listView2 import ftListWidget
from SLM.vector_db.vector_db import VectorDB
from applications.collectionTools.components.event_dispatcher import EventDispatcher
from applications.collectionTools.views.collection_list_view.collectionView.listViewModel import fsTypeConverter, \
    lwItemTemplate
from applications.collectionTools.views.collection_list_view.selectionManager.selection_manager import SelectionManager


@singleton
class SimilarItemsExplorerModel:
    def __init__(self):
        # todo add image search mode fore images mean
        # todo add mode self select
        self.self_select_handle = False
        self.disabled = False
        self.list_view = ftListWidget()
        self.list_view.data_list_cursor.data_converter = fsTypeConverter()
        self.list_view.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        SelectionManager().on_selection_changed.append(self.on_selection_changed)
        path = r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties\not sorted"
        self.scope_query = {"local_path": {"$regex": re.escape(path)}}
        if not self.disabled:
            # on create show dialog on create scope
            self.vector_table = VectorDB.get_pref("FileRecordResnet50")
            self.scope = self.vector_table.get_search_scope(self.scope_query, SearchScopeMongoDb)
        self.list_view.set_listener_on_selected(self.on_selected)
        self.search_image_mode = "first"

    def on_selected(self, *args, **kwargs):
        kwargs.setdefault("source", self)
        kwargs.setdefault("mode", self.list_view.selection_mode)
        EventDispatcher().dispatch_event("on_select", *args, **kwargs)

    def on_selection_changed(self, selection, *args, **kwargs):
        # todo prevent multiple call bay use bgWorker
        # todo implement more complex protokol for selectd items each selected itemss have additional data
        if kwargs.get("source", None) is self:
            return
        if self.disabled:
            return
        self.list_view.data_list.clear()
        if len(selection) == 0:
            return
        append_list = []

        for file in selection:
            first: FileRecord = file
            # todo implement app lock or time of loading search index
            search_ind = self.scope
            result = search_ind.search(first.get_record_data(), 10)
            for res in result.results:
                record = FileRecord(res.data_item["_id"])
                append_list.append(record)
        append_list = list(set(append_list))

        for record in append_list:
            # todo implement extend method
            self.list_view.data_list.append(record)

    def refreh_scope(self):
        self.vector_table = VectorDB.get_pref("FileRecordResnet50")
        self.scope = self.vector_table.get_search_scope(self.scope_query, SearchScopeMongoDb)
        self.on_selection_changed(SelectionManager().get_selection())

    def set_path(self, path):
        self.scope_query = {"local_path": {"$regex": re.escape(path)}}
        self.refreh_scope()