import io
import os
import re
import json
from collections import defaultdict
from copy import copy
from enum import Enum
from typing import List

import PySide6
import loguru
from PySide6.QtCore import Qt
from pydantic import BaseModel, Field

from PySide6.QtWidgets import QDialog, QWidget

from tqdm import tqdm
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.core import Allocator, Event
from SLM.appGlue.selectionManager import SelectionManager, SelectionManagerUser
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.groupcontext import group
from SLM.mongoext.mongoQueryTextParser2 import parse_mongo_query_with_normalization
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QMenu, QToolButton, \
    QCheckBox

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout, QTMessages
from SLM.pySide6Ext.binding import binding_load
from SLM.pySide6Ext.widgets.ImageWidget import ImageOverlayWidget
from SLM.pySide6Ext.widgets.tools import WidgetBuilder
from applications.image_graph_qt.db_maintain_panel_v2 import DBMaintainView

binding_load()


# todo add context menu move to folder of selected use selection manager first selected item move on context executed item to folder of first selected


# todo add hailathing folders with red hailaith child same part of path of parent folder and add posibility hailaith by green desired string in fpath of item end relation item set in setings menu end persistent

def update_relation_symmetrically(relation: RelationRecord, new_sub_type: str):
    """
    Updates a relation's sub_type and its symmetrical counterpart.
    """
    # 1. Update the original relation
    relation.set_field_val("sub_type", new_sub_type)
    relation.onEdit.fire()
    # 2. Find the reverse relation
    reverse_relation = RelationRecord.find_one({
        "from_id": relation.to_id,
        "to_id": relation.from_id,
        "type": relation.type
    })

    # 3. If the reverse relation exists, update it too
    if reverse_relation:
        reverse_relation.set_field_val("sub_type", new_sub_type)
        reverse_relation.onEdit.fire()
        loguru.logger.info(f"Updated symmetrical relation for {reverse_relation._id}")


def progress_sorted(iterable, key=None, reverse=False, delay=0.1):
    arr = list(iterable)
    pbar = tqdm(total=len(arr), desc="Sorting Progress", unit="item")

    def wraped_key(item):
        pbar.update(1)
        return key(item) if key else item

    arr.sort(key=wraped_key, reverse=reverse)
    return arr


class SettingsDialog(QDialog):
    def __init__(self, settings: "FilterSettings", parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")

        V_layout = QVBoxLayout()
        self.setLayout(V_layout)

        # Filter Panel
        filter_panel = QVBoxLayout()

        # Row 1: Thresholds
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Threshold Min:"))
        self.threshold_min_edit = QLineEdit(str(self.settings.threshold_min))
        row1.addWidget(self.threshold_min_edit)

        row1.addWidget(QLabel("Threshold Max:"))
        self.threshold_max_edit = QLineEdit(str(self.settings.threshold_max))
        row1.addWidget(self.threshold_max_edit)
        filter_panel.addLayout(row1)

        # Row 4: Subtypes
        row4 = FlowLayout()
        self.subtype_checkboxes = {}
        for subtype in ImagaeSearchRelSubType:
            checkbox = QCheckBox(subtype.value)
            checkbox.setChecked(subtype.value in self.settings.subtypes)
            checkbox.stateChanged.connect(lambda state, s=subtype.value: self.update_subtypes(s, state))
            self.subtype_checkboxes[subtype.value] = checkbox
            row4.addWidget(checkbox)
        filter_panel.addLayout(row4)

        V_layout.addLayout(filter_panel)

        # OK and Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        V_layout.addLayout(button_layout)

    def update_subtypes(self, subtype, state):
        if state == Qt.Checked.value and subtype not in self.settings.subtypes:
            self.settings.subtypes.append(subtype)
        elif state == Qt.Unchecked.value and subtype in self.settings.subtypes:
            self.settings.subtypes.remove(subtype)

    def accept(self):
        try:
            self.settings.threshold_min = float(self.threshold_min_edit.text())
            self.settings.threshold_max = float(self.threshold_max_edit.text())
            self.settings.save()
        except ValueError:
            # Optionally, show an error message to the user
            pass
        super().accept()


#todo not delete relation marck as wrong
#todo improwe wrong relation edit

class Similar_graph_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        WidgetBuilder.add_menu_item(menu, "select items", lambda x: self.sel_items())
        WidgetBuilder.add_menu_item(menu, "Create ref on selected", lambda x: self.create_ref_on_sel())
        WidgetBuilder.add_menu_item(menu, "Merge groups", lambda x: self.merge_groups())
        WidgetBuilder.add_menu_item(menu, "Simplify relations",
                                    lambda x: self.simplifyrels())
        WidgetBuilder.add_menu_item(menu, "Delete selected files", lambda x: self.delete_selected_files())
        WidgetBuilder.add_menu_item(menu, "Delete brocken relations", lambda x: self.delete_broken_rels())
        WidgetBuilder.add_menu_item(menu, "DB Maintenance", lambda x: self.open_db_maintenance())
        WidgetBuilder.add_menu_item(menu, "Settings...", lambda x: self.open_settings_dialog())

    def sel_items(self):
        """
        Selects items in the current view based on the selection manager.
        """
        selection_manager: SelectionManager = Allocator.get_instance(RelationEditView).selection_man
        for user in selection_manager.selectionUsers:
            if isinstance(user, lwItemTemplate):
                user.set_selected(True)

    def open_settings_dialog(self):
        view = Allocator.get_instance(RelationEditView)
        dialog = SettingsDialog(view.settings, self._main_window)
        dialog.exec()

    def open_db_maintenance(self):
        dialog = QDialog(self._main_window)
        dialog.setWindowTitle("DB Maintenance")
        layout = QVBoxLayout()
        db_maintain_view = DBMaintainView()
        layout.addWidget(db_maintain_view)
        dialog.setLayout(layout)
        dialog.exec()

    def delete_broken_rels(self):
        view:RelationEditView = Allocator.get_instance(RelationEditView)
        filter = view.rel_filter
        query = filter.get_filter()
        relations = RelationRecord.find(query)
        for relation in tqdm(relations):
            from_rec: FileRecord = FileRecord.get_by_id(relation.from_id)
            to_rec: FileRecord = FileRecord.get_by_id(relation.to_id)
            if from_rec is None or to_rec is None:
                relation.delete_rec()
                continue
            if not os.path.exists(from_rec.full_path):
                relation.delete_rec()
                from_rec.delete_rec()

            if not os.path.exists(to_rec.full_path):
                relation.delete_rec()
                to_rec.delete_rec()

            if from_rec.full_path == to_rec.full_path:
                relation.delete_rec()
                if from_rec.id != to_rec.id:
                    to_rec.delete_rec()
                continue

    def delete_selected_files(self):
        selection_manager: SelectionManager = Allocator.get_instance(RelationEditView).selection_man
        sel = selection_manager.get_selection_as()
        if len(sel) > 0:
            for item in tqdm(sel):
                item: FileRecord
                item.delete()
        #refresh list view
        selection_manager.clear_selection()
        listv: ListViewWidget = Allocator.get_instance(RelationEditView).list_view_etalons
        listv.refresh()
        print("refreshed")

    def create_ref_on_sel(self, *args, **kwargs):
        print("Create ref on sel")
        selection_manager: SelectionManager = Allocator.get_instance(RelationEditView).selection_man
        sel = selection_manager.get_selection_as()
        if len(sel) > 1:
            first = sel[0]
            other = sel[1:]
            for items in other:
                rel = RelationRecord.set_relation(first, items, "similar_search")
                rel.set_field_val("sub_type", ImagaeSearchRelSubType.manual.value)
                rel.set_field_val("distance", 0.99)
                rel.set_field_val("emb_type", "manual")
        selection_manager.clear_selection()

    def merge_groups(self, *args, **kwargs):
        print("Create ref on sel")
        selection_manager: SelectionManager = Allocator.get_instance(RelationEditView).selection_man
        sel = selection_manager.get_selection_as()
        if len(sel) > 1:
            all_rel = []
            first = sel[0]
            other = sel[1:]
            for items in other:
                all_rel.append(items)
                get_oter_rels = RelationRecord.find({"from_id": items._id, "type": "similar_search"})
                other_rels = list(get_oter_rels)
                file_recs = [FileRecord.get_by_id(x.to_id) for x in other_rels]
                all_rel.extend(file_recs)
                for rel in other_rels:
                    rel.delete_rec()
            for item in all_rel:
                rel = RelationRecord.set_relation(first, item, "similar_search")
                rel.set_field_val("sub_type", ImagaeSearchRelSubType.manual.value)
                rel.set_field_val("distance", 0.99)
                rel.set_field_val("emb_type", "manual")
        selection_manager.clear_selection()

    def simplifyrels(self, *args, **kwargs):
        print("Create ref on sel")

        # Загружаем все связи с типом "similar_search"
        rels = RelationRecord.find({"type": "similar_search", "sub_type": ImagaeSearchRelSubType.none.value})

        # Оптимизируем поиск дублирующих связей
        processed = set()  # Для отслеживания уже обработанных пар
        for rel in tqdm(rels):
            source = FileRecord.get_by_id(rel.from_id)

            # Получаем все связи для текущего source
            targets_rel = list(RelationRecord.find(
                {"from_id": source._id, "type": "similar_search", "sub_type": ImagaeSearchRelSubType.none.value}))

            # Используем defaultdict для группировки связей по to_id
            relations_map = defaultdict(list)
            for relx in targets_rel:
                relations_map[relx.to_id].append(relx)

            # Проверяем и удаляем дублирующие связи
            for relx in targets_rel:
                for rely in targets_rel:
                    if relx.to_id != rely.to_id:  # Исключаем одну и ту же связь
                        pair = (relx.to_id, rely.to_id)
                        back_pair = (rely.to_id, relx.to_id)

                        if pair in processed or back_pair in processed:
                            continue  # Пропускаем уже обработанные пары

                        # Проверка прямой и обратной связи
                        rel_rec = RelationRecord.find_one(
                            {"from_id": relx.to_id, "to_id": rely.to_id, "type": "similar_search"})
                        if rel_rec:
                            rel_rec.set_field_val("sub_type", ImagaeSearchRelSubType.hiden.value)

                        back_rel = RelationRecord.find_one(
                            {"from_id": rely.to_id, "to_id": relx.to_id, "type": "similar_search"})
                        if back_rel:
                            back_rel.set_field_val("sub_type", ImagaeSearchRelSubType.hiden.value)

                        # Добавляем пары в обработанные
                        processed.add(pair)
                        processed.add(back_pair)


class FilterSettings(BaseModel):
    threshold_min: float = Field(default=0.0)
    threshold_max: float = Field(default=1.0)
    subtypes: List[str] = Field(default_factory=lambda: [x.value for x in ImagaeSearchRelSubType])
    text_filter: str = ""

    def save(self, path="image_graph_config.json"):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, path="image_graph_config.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                return cls.model_validate_json(f.read())
        return cls()


class ImagaeSearchRelSubType(Enum):
    wrong = "wrong"
    similar = "similar"
    not_similar = "near_dub"
    similar_style = "similar_style"
    manual = "manual"
    some_person = "some_person"
    some_image_set = "some_image_set"
    other = "other"
    hiden = "hiden"
    none = "none"


class ImageSearchRelFilter:
    def __init__(self, settings: FilterSettings):
        self.settings = settings
        self.query = {}

    def get_filter(self):
        return {"distance": {"$gt": self.settings.threshold_min,
                             "$lt": self.settings.threshold_max},
                "type": "similar_search",
                "sub_type": {"$in": self.settings.subtypes},
                }

    def get_pipline_filter(self):
        return {"relations.distance": {"$gt": self.settings.threshold_min, "$lt": self.settings.threshold_max},
                "relations.type": "similar_search",
                "relations.sub_type": {"$in": self.settings.subtypes},
                }

    def escape(self, s, escapechar, specialchars):
        return "".join(escapechar + c if c in specialchars or c == escapechar else c for c in s)

    def recursive_postprocess_filter(self, filter, parent_key=None):
        if isinstance(filter, dict):
            for key, value in filter.items():
                if key == "$regex" and parent_key == "local_path":
                    filter[key] = "^" + re.escape(value)
                elif key == "$in":
                    filter[key] = [self.recursive_postprocess_filter(v) for v in value]
                elif key == "$or":
                    filter[key] = [self.recursive_postprocess_filter(v) for v in value]
                elif key == "$and":
                    filter[key] = [self.recursive_postprocess_filter(v) for v in value]
                elif key == "$not":
                    filter[key] = self.recursive_postprocess_filter(value)
                elif key == "$elemMatch":
                    filter[key] = self.recursive_postprocess_filter(value)
                else:
                    filter[key] = self.recursive_postprocess_filter(value, key)
        elif isinstance(filter, list):
            for i, value in enumerate(filter):
                filter[i] = self.recursive_postprocess_filter(value, parent_key)
        return filter

    def parse_query(self):
        if not self.settings.text_filter:
            self.query = {}
            return
        try:
            text_filter = self.escape(self.settings.text_filter, "\\", [])

            self.query = parse_mongo_query_with_normalization(text_filter)
            self.query = self.recursive_postprocess_filter(self.query)
        except:
            # convert to search
            self.query = {"$text": {"$search": self.settings.text_filter}}
            try:
                FileRecord.find(self.query)
            except:
                self.query = {}

    def get_pipline(self):
        self.parse_query()
        pipeline = [
            # Stage 1: Match file records by local_path regex
            {
                "$match":
                    self.query

            },
            # Stage 2: Lookup relations collection
            {
                "$lookup": {
                    "from": "relation_records",
                    "localField": "_id",
                    "foreignField": "from_id",
                    "as": "relations"
                }
            },
            # Stage 3: Filter relations with rel_filter
            {
                "$unwind": {
                    "path": "$relations",
                    "preserveNullAndEmptyArrays": False  # Optional
                }
            },
            # Stage 4: Match relations by distance
            {
                "$match": self.get_pipline_filter()
            },
            # Stage 4: Group by _id
            {
                "$group": {
                    "_id": "$_id",
                    "relations": {"$push": "$relations"}
                }
            },
            # filter relations list not empty
            {
                "$match": {
                    "relations": {"$ne": []}
                }
            },
            # sort by name
            {
                "$sort": {"name": 1}
            }
        ]
        return pipeline

    def is_subtype_in_filter(self, subtype: ImagaeSearchRelSubType):
        return subtype.value in self.settings.subtypes


class RelationEditView(PySide6GlueWidget):

    def __init__(self):
        self.settings = FilterSettings.load()
        self.rel_filter = ImageSearchRelFilter(self.settings)
        self.list_view_etalons: ListViewWidget = None
        Allocator.res.register(self)
        self.window_size = None
        MessageSystem.Subscribe(QTMessages.MAIN_WINDOW_RESIZED, self, self.on_main_window_resized)
        self.selection_man = SelectionManager()
        super().__init__()

    def on_main_window_resized(self, size):
        self.window_size = size

    def build_list_view(self, *args, **kwargs):
        self.settings.save()
        self.list_view_etalons.data_list.clear()
        self.selection_man.selectionUsers.clear()

        pipeline = self.rel_filter.get_pipline()
        res = FileRecord.collection().aggregate(pipeline)
        res_list = list(res)
        add_records = []
        for rec in tqdm(res_list):
            file = FileRecord.get_by_id(rec["_id"])
            add_records.append(file)

        self.list_view_etalons.data_list.extend(add_records)
        self.list_view_etalons.list_update_metric()

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        self.setLayout(V_layout)

        # Filter Panel
        filter_panel = QHBoxLayout()

        filter_panel.addWidget(QLabel("Text Filter:"))
        self.text_filter_edit = QLineEdit(self.settings.text_filter)
        self.text_filter_edit.textChanged.connect(self.update_text_filter)
        filter_panel.addWidget(self.text_filter_edit)

        # Find Button
        find_button = QPushButton("Find")
        find_button.clicked.connect(self.build_list_view)
        filter_panel.addWidget(find_button)

        V_layout.addLayout(filter_panel)

        # List View
        self.list_view_etalons = ListViewWidget()
        self.list_view_etalons.selection_mode = "multi"
        self.list_view_etalons.data_list_cursor.items_per_page = 30
        self.list_view_etalons.data_list_cursor.sort_alg["by_path"] = lambda lst: sorted(lst,
                                                                                         key=lambda x: x.local_path)
        self.list_view_etalons.data_list_cursor.sort = "by_path"
        self.list_view_etalons.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        FileRecord.onDeleteGlobal += self.on_file_delete
        V_layout.addWidget(self.list_view_etalons)

    def update_text_filter(self, text):
        self.settings.text_filter = text

    def on_file_delete(self, file_record: FileRecord):
        if file_record in self.list_view_etalons.data_list:
            self.list_view_etalons.data_list.remove(file_record)


class ListRelItemTemplate(ListViewItemWidget, SelectionManagerUser):
    target_file: FileRecord
    source_file: FileRecord
    rel_vi_type: str = "in"  # or "out"

    def __init__(self, *args, **kwargs):
        ListViewItemWidget.__init__(self, **kwargs)
        SelectionManagerUser.__init__(self)

    def register(self):
        manager = Allocator.get_instance(RelationEditView).selection_man
        self.parent_manager = manager
        manager.register_user(self)

        if self.data_context.vi_type == "out":
            target = FileRecord.get_by_id(self.data_context.to_id)
            source_file = FileRecord.get_by_id(self.data_context.from_id)
            self.rel_vi_type = "out"
        else:
            target = FileRecord.get_by_id(self.data_context.from_id)
            source_file = FileRecord.get_by_id(self.data_context.to_id)
            self.rel_vi_type = "in"
        self.selection_data.selection = self.target_file = target
        self.source_file = source_file
        FileRecord.onDeleteGlobal += self.on_file_delete
        self.data_context.onEdit += self.load_data

    def dispose(self):
        try:
            super().dispose()
            manager = Allocator.get_instance(RelationEditView).selection_man
            manager.unregister_user(self)
            self.sel_checkbox.stateChanged.disconnect()

        except Exception as e:
            loguru.logger.error(e)
        try:
            FileRecord.onDeleteGlobal -= self.on_file_delete
            self.data_context.onEdit -= self.load_data
        except Exception as e:
            loguru.logger.error(f"Error removing onDelete handler: {e}")

    def on_file_delete(self, obj):
        if obj in self.parent_list_view.data_list:
            self.parent_list_view.data_list.remove(obj)

    def build_header(self):
        self.mose_event_propagate = False

        vert_layout = QVBoxLayout()
        self.content.addLayout(vert_layout)

        self.sel_checkbox = QCheckBox()
        self.name_label = QLabel()
        self.name_label.setWordWrap(True)
        vert_layout.addWidget(self.name_label)

        tools_panel = FlowLayout()
        tools_panel.addWidget(self.sel_checkbox)
        vert_layout.addLayout(tools_panel)

        with group():
            rem_rel_button = QPushButton("X rel")
            rem_rel_button.setFixedSize(25, 25)
            rem_rel_button.setStyleSheet("font-size: 10px;")
            rem_rel_button.setToolTip("Remove relation")
            rem_rel_button.clicked.connect(self.removeRelation)
            tools_panel.addWidget(rem_rel_button)

        self.rel_subtype_label = QLabel()
        vert_layout.addWidget(self.rel_subtype_label)
        self.image_w = ImageOverlayWidget(parent=self)
        vert_layout.addWidget(self.image_w)
        self.reit_label = QLabel()
        self.reit_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128);" "color: white;" "padding: 5px;" "font-size: 10px;")
        self.image_w.get_layout().addWidget(self.reit_label)

        self.register()

        self.selection_handler = lambda x: self.set_selected(x)
        self.sel_checkbox.stateChanged.connect(self.selection_handler)

        self.load_data()

    def load_data(self):
        data = {}
        relation: RelationRecord = self.data_context
        data["distance"] = relation.get_field_val("distance")
        data["euclidean"] = relation.get_field_val("euclidean")
        data["manhattan"] = relation.get_field_val("manhattan")
        data["hamming"] = relation.get_field_val("hamming")
        data["dot"] = relation.get_field_val("dot")
        data["emb_type"] = relation.get_field_val("emb_type")
        data["sub_type"] = relation.get_field_val("sub_type")
        try:
            data["rating"] = self.target_file.rating

            self.thumb_path = self.target_file.get_thumb("medium")
        except Exception as e:
            loguru.logger.error(e)
            data["rating"] = "no file"
            self.thumb_path = None
        self.set_gui(data)

    @PySide6.QtCore.Slot(object)
    def set_gui(self, data):
        try:
            self.name_label.setText(str(self.target_file.name))
            hint_text = (
                f"id: {str(self.target_file._id)}\n"
                f"Relation type: {self.rel_vi_type}\n"
                f"Rel sub type: {self.data_context.get_field_val('sub_type')}\n"
                f"Emb type: {data['emb_type']}\n"
                f"ang:{data['distance']} euclidean:{data['euclidean']} manhattan:{data['manhattan']} hamming:{data['hamming']} dot:{data['dot']}"
            )
            rel_subtype = data["sub_type"]
            self.rel_subtype_label.setText(str(rel_subtype))
            self.image_w.load_image(self.thumb_path)
            self.image_w.setToolTip(hint_text)
            self.reit_label.setText(str(data["rating"]))
        except Exception as e:
            loguru.logger.error(e)

    def set_item_sub_type(self, sub_type):
        if self.data_context is None:
            loguru.logger.error("data_context is None, cannot set sub_type")
            return
        update_relation_symmetrically(self.data_context, sub_type)
        self.rel_subtype_label.setText(str(sub_type))

    def on_set_selected(self, state):
        if self.sel_checkbox.checkState() == Qt.Checked:
            self.set_selected(True)
        else:
            self.set_selected(False)

    def delete_file(self):
        file_record = self.target_file
        file_record.delete()
        file_rel: RelationRecord = self.data_context
        file_rel.delete_rec()

    def set_selected(self, selected: bool):
        try:
            if selected:
                self.sel_checkbox.setCheckState(Qt.Checked)
            else:
                self.sel_checkbox.setCheckState(Qt.Unchecked)
            self.sel_checkbox.update()
        except Exception as e:
            print(e)
        super().set_selected(selected)

    def removeRelation(self):
        rel_filter = Allocator.get_instance(RelationEditView).rel_filter
        parent_lv = self.parent().parent_list

        file_rel: RelationRecord = self.data_context
        update_relation_symmetrically(file_rel, ImagaeSearchRelSubType.wrong.value)
        if not rel_filter.is_subtype_in_filter(ImagaeSearchRelSubType.wrong):
            parent_lv.data_list.remove(self.data_context)

    def delete_relation(self):
        file_rel: RelationRecord = self.data_context
        file_rel.delete_rec()

    def get_group(self, group_param):
        directory = self.target_file.local_path
        return directory

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("refresh thumbnail", self.refresh_thumbnail)
        menu.addAction("Open in explorer", self.open_in_explorer)
        menu.addAction("Delete File", self.delete_file)
        menu.addAction("Delete relation", self.delete_relation)
        menu.addAction("Delete File record", self.delete_file_record)
        submenu = menu.addMenu("Set relation type")
        for rel_type in ImagaeSearchRelSubType:
            submenu.addAction(rel_type.value, lambda x=rel_type.value: self.set_item_sub_type(x))
        menu.exec_(event.globalPos())

    def refresh_thumbnail(self):
        self.target_file.refresh_thumb()
        self.load_data()

    def delete_file_record(self):
        file_record: FileRecord = self.target_file
        file_record.delete_rec()
        self.parent_list_view.refresh()

    def open_in_explorer(self):
        os.system(f'explorer /select,"{self.target_file.full_path}"')


class lwItemTemplate(ListViewItemWidget, SelectionManagerUser):
    list_view_rel: ListViewWidget
    image_w: ImageOverlayWidget
    file: FileRecord

    def __init__(self, *args, **kwargs):
        ListViewItemWidget.__init__(self, **kwargs)
        SelectionManagerUser.__init__(self)

    def register(self):
        self.file: FileRecord = self.data_context
        self.setAcceptDrops(True)
        selection_man = Allocator.get_instance(RelationEditView).selection_man
        selection_man.register_user(self)
        self.selection_data.selection = self.file
        MessageSystem.Subscribe(QTMessages.MAIN_WINDOW_RESIZED, self, self.on_main_window_resized)

    def dispose(self):
        MessageSystem.Unsubscribe(QTMessages.MAIN_WINDOW_RESIZED, self)
        selection_man = Allocator.get_instance(RelationEditView).selection_man
        selection_man.unregister_user(self)

    def build_header(self):
        self.register()

        self.horiz_layout = QHBoxLayout()
        self.content.addLayout(self.horiz_layout)

        col1_container = QWidget()
        self.vert1 = QVBoxLayout(col1_container)
        col1_container.setFixedWidth(300)

        self.vert2 = QVBoxLayout()

        self.horiz_layout.addWidget(col1_container)
        self.horiz_layout.addLayout(self.vert2)
        self.name_label = (
            WidgetBuilder(QLabel(str(self.file.name)))
            .set_word_wrap(True)
            .add_to_layout(self.vert1)
        ).build()
        path = self.file.local_path
        self.path_label = (WidgetBuilder(QLabel(str(path)))
                           .set_word_wrap(True)
                           .add_to_layout(self.vert1)
                           ).build()

        image_w = ImageOverlayWidget(parent=self)
        self.image_w = image_w
        self.vert1.addWidget(image_w)

        layout = image_w.get_layout()
        # set layout semi transparent black
        reit_label = QLabel(str(self.file.rating))
        self.reit_label = reit_label
        # set style
        reit_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128);" "color: white;" "padding: 5px;" "font-size: 10px;")
        layout.addWidget(reit_label)
        with group():
            tools_panel = FlowLayout()

            self.sel_checkbox = QCheckBox()
            self.sel_handler = lambda x: self.on_set_selected(x)
            self.sel_checkbox.stateChanged.connect(self.sel_handler)
            tools_panel.addWidget(self.sel_checkbox)

            rem_rel_button = QPushButton("X rel")
            rem_rel_button.setFixedSize(25, 25)
            rem_rel_button.setStyleSheet("font-size: 10px;")
            #set hint
            rem_rel_button.setToolTip("Remove all relations")

            rem_rel_button.clicked.connect(lambda: self.removeRelation())
            tools_panel.addWidget(rem_rel_button)

            for rel_type in ImagaeSearchRelSubType:
                t_button = QPushButton(rel_type.value)
                t_button.clicked.connect(lambda a, x=rel_type.value: self.set_all_item_sub_type(x))
                tools_panel.addWidget(t_button)

        #BGWorker.instance().add_task(LoadImageTask(image_w, thumbnail_path))

        self.list_view_rel = ListViewWidget()
        self.list_view_rel.template.itemTemplateSelector.add_template(RelationRecord, ListRelItemTemplate)
        self.list_view_rel.selection_mode = "multi"
        self.vert2.addWidget(self.list_view_rel)
        view = Allocator.get_instance(RelationEditView)
        size = view.window_size
        self.list_view_rel.setFixedWidth(size.width() - 320)
        self.list_view_rel.setFixedHeight(450)
        self.vert2.addLayout(tools_panel)
        self.load_data()

    def load_data(self):
        view = Allocator.get_instance(RelationEditView)
        # get out relations
        rel_filter = view.rel_filter.get_filter()
        rel_filter["from_id"] = self.file._id
        rel_list = []
        r = RelationRecord.find(rel_filter)
        for rel in r:
            rel.vi_type = "out"
            rel_list.append(rel)

        self.rel_list = rel_list

        self.thumbnail_path = self.file.get_thumb("medium")
        self.set_gui()

    def set_gui(self):
        self.image_w.load_image(self.thumbnail_path)
        self.image_w.setToolTip(str(self.file._id))
        if len(self.rel_list) > 0:
            self.list_view_rel.data_list.extend(self.rel_list)

    def set_rel_all_in_folder(self):
        file = self.data_context
        folder_files = FileRecord.find({"local_path": {"$regex": f"^{re.escape(file.local_path)}"}})
        for other_file in tqdm(folder_files):
            for file_2 in folder_files:
                if file_2 != other_file:
                    rel = RelationRecord.find(
                        {"$or": [{"from_id": other_file._id, "to_id": file_2._id, "type": "similar_search"},
                                 {"from_id": file_2._id, "to_id": other_file._id, "type": "similar_search"}]})
                    for r in rel:
                        r.set_field_val("sub_type", ImagaeSearchRelSubType.manual.value)

        self.parent_list_view.refresh()

    def reload_child_data(self):
        items_widgets = self.list_view_rel.items_nodes.values()
        for item in items_widgets:
            item.load_data()

    def delete_file(self):
        file_record = self.data_context
        file_record.delete()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                file = FileRecord.get_record_by_path(path)
                if file is not None:
                    rel = RelationRecord.set_relation(self.selection_data.selection, file, "similar_search")
                    rel.set_field_val("sub_type", ImagaeSearchRelSubType.manual.value)
                    rel.set_field_val("distance", 0.1)
                    rel.set_field_val("emb_type", "manual")
                    rel.vi_type = "out"
                    self.list_view_rel.data_list.append(rel)
            else:
                print(f"Path is not file {path}")
        event.acceptProposedAction()

    def set_all_item_sub_type(self, sub_type):
        for rel in self.list_view_rel.data_list_cursor.get_filtered_data():
            update_relation_symmetrically(rel, sub_type)
        self.reload_child_data()

    def incrise_rel_level(self):
        child_rels = self.list_view_rel.data_list_cursor.get_filtered_data()
        rel_filter = Allocator.get_instance(RelationEditView).rel_filter
        targets = []
        for rel in child_rels:
            if rel.vi_type == "out":
                target = FileRecord.get_by_id(rel.to_id)
            else:
                target = FileRecord.get_by_id(rel.from_id)
            if target not in targets:
                targets.append(target)
        adition_items = []
        for target in targets:
            # extract all relations
            rel_filter_copy = copy(rel_filter.get_filter())
            rel_filter_copy["from_id"] = target._id
            r = RelationRecord.find(rel_filter_copy)
            for rel in r:
                rel.vi_type = "out"
            adition_items.extend(r)
            rel_filter_copy = copy(rel_filter.get_filter())
            rel_filter_copy["to_id"] = target._id
            r = RelationRecord.find(rel_filter_copy)
            for rel in r:
                rel.vi_type = "in"
            adition_items.extend(r)
        self.list_view_rel.data_list.extend(adition_items)

    def set_reit(self, rating):
        file = self.data_context
        file.rating = rating
        self.reit_label.setText(str(rating))

    def on_set_selected(self, state):
        if self.sel_checkbox.checkState() == Qt.Checked:
            self.set_selected(True)
        else:
            self.set_selected(False)

    def set_selected(self, selected: bool):
        try:
            if selected:
                self.sel_checkbox.setCheckState(Qt.Checked)
            else:
                self.sel_checkbox.setCheckState(Qt.Unchecked)
            self.sel_checkbox.update()
        except Exception as e:
            print(e)
        super().set_selected(selected)

    def on_main_window_resized(self, size):
        #change width of relation list
        self.list_view_rel.setFixedWidth(size.width() - 320)

    def removeRelation(self):
        all_child_rels = self.list_view_rel.data_list_cursor.get_filtered_data()
        selected_rels = self.list_view_rel.get_selected_items()
        rl = self.list_view_rel.data_list_cursor.get_filtered_data()
        for r in all_child_rels:
            if r not in selected_rels:
                r.set_field_val("sub_type", ImagaeSearchRelSubType.wrong.value)
                self.list_view_rel.data_list.remove(r)
        self.list_view_rel.clear_selection()

    def set_child_relation_sub_type_deep(self, sub_type):
        child_widgets = list(self.list_view_rel.items_nodes.values())
        target_files = [w.target_file for w in child_widgets]

        # Update the direct relations shown in the list
        for widget in child_widgets:
            widget.set_item_sub_type(sub_type)

        # Iterate through pairs and update relations between them
        for i in range(len(target_files)):
            for j in range(i + 1, len(target_files)):
                file1_id = target_files[i]._id
                file2_id = target_files[j]._id

                # Find relations between file1 and file2
                related_rels = RelationRecord.find({
                    "type": "similar_search",
                    "$or": [
                        {"from_id": file1_id, "to_id": file2_id},
                        {"from_id": file2_id, "to_id": file1_id}
                    ]
                })

                for rel in related_rels:
                    # We can call the symmetrical update here as well to be safe,
                    # it will just re-set the value if already done.
                    update_relation_symmetrically(rel, sub_type)

        self.reload_child_data()
        self.parent_list_view.refresh()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("refresh thumbnail", self.refresh_thumbnail)
        menu.addAction("Open in explorer", lambda: os.system(f'explorer /select,"{self.file.full_path}"'))
        menu.addAction("Increase rel level", lambda: self.incrise_rel_level())
        menu.addAction("Delete File", lambda: self.delete_file())
        menu.addAction("Set related all in folder", lambda: self.set_rel_all_in_folder())

        rating_submenu = menu.addMenu("Set rating")
        for i in range(1, 6):
            rating_submenu.addAction(str(i), lambda x=i: self.set_reit(x))

        rel_type_submenu = menu.addMenu("Set all child relation type")
        for rel_type in ImagaeSearchRelSubType:
            rel_type_submenu.addAction(rel_type.value,
                                       lambda x=rel_type.value: self.set_all_item_sub_type(x))

        menu.addSeparator()

        deep_rel_menu = menu.addMenu("Mark Child Relations Deep")
        for rel_type in ImagaeSearchRelSubType:
            deep_rel_menu.addAction(rel_type.value, lambda x=rel_type.value: self.set_child_relation_sub_type_deep(x))

        menu.exec_(event.globalPos())

    def refresh_thumbnail(self):
        self.file.refresh_thumb()
        self.load_data()


config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"

QtApp = Similar_graph_app()
QtApp.set_main_widget(RelationEditView())

QtApp.run()
