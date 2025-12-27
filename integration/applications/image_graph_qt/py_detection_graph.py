import os
import re
from copy import copy
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers

from tqdm import tqdm

from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.core import Allocator
from SLM.appGlue.selectionManager import SelectionManager, SelectionManagerUser
from SLM.files_data_cache.pool import PILPool
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.object_recognition.object_recognition import Detection, DetectionObjectClass, Recognized_object
from SLM.groupcontext import group
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QMenu, QSizePolicy, QToolButton, \
    QCheckBox

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout, pil_to_pixmap, QTMessages
from SLM.pySide6Ext.binding import binding_load
from SLM.pySide6Ext.widgets.obj_detection.edit_rec_obj import EditRecognizedObjectsDialog
from SLM.pySide6Ext.widgets.tools import WidgetBuilder as wb

binding_load()


class DetectionSearchRelSubType(Enum):
    wrong = "wrong"
    similar = "similar"
    not_similar = "near_dub"
    similar_style = "similar_style"
    manual = "manual"
    some_person = "some_person"
    other = "other"
    none = "none"


class SimSearchRelation:
    def __init__(self, rel_record):
        self.rel_record = rel_record
        self.from_detection = Detection(rel_record.from_id)
        self.to_detection = Detection(rel_record.to_id)
        self.distance = rel_record.distance
        self.sub_type = DetectionSearchRelSubType(rel_record.sub_type)
        self.search_class = DetectionObjectClass(rel_record.search_class_id)


# add reit display overlay on image
# add self ref display overlay on image
# add dialog find faces (edit faces by search query)



class Similar_graph_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        # set window size
        self._main_window.resize(1200, 800)
        #set window title
        self._main_window.setWindowTitle("Similarity graph")

    def edit_recognized_objects(self):
        dialog = EditRecognizedObjectsDialog("Edit recognized dictionary", self._main_window)
        dialog.exec()

    def optimize_face_relations(self):
        face_relations = RelationRecord.find({"type": "similar_obj_search"})
        face_relations = list(face_relations)
        # todo:optimize face graph


class Similar_gr_system:
    def __init__(self):
        self.rel_filter = ImageSearchRelFilter()
        self.items = []

    def get_detections(self):
        pipeline = self.rel_filter.get_pipline()
        res = FileRecord.collection().aggregate(pipeline)
        res_list = list(res)
        show_detections = []
        for rec in tqdm(res_list):
            for rel in rec["relations"]:
                detection = rel["from_id"]
                detection = Detection(detection)
                show_detections.append(detection)
        show_detections = list(set(show_detections))
        # filtrate
        for detection in copy(show_detections):
            query = copy(self.rel_filter.get_filter())
            query["from_id"] = detection._id
            out_relations = RelationRecord.find(query)
            if len(out_relations) == 0:
                show_detections.remove(detection)
        return show_detections

    def set_threshold(self, min, max):
        self.rel_filter.threshold_min = min
        self.rel_filter.threshold_max = max

    def set_working_folder(self, folder):
        self.rel_filter.working_folder = folder


class Image_graph_app_bind(PropUser):
    working_folder = PropInfo(default=r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties")
    threshold_min = PropInfo(default=0.0)
    threshold_max = PropInfo(default=1.0)


class ImageSearchRelFilter:
    def __init__(self, working_folder=r"E:\rawimagedb\repository"):
        self.threshold_min = 0.0
        self.threshold_max = 1.0
        self.file_reit_min = 0.0
        self.file_reit_max = 1.0
        self.working_folder = working_folder
        self.current_detection_class = "face"
        self.subtypes = [DetectionSearchRelSubType.none]

    def get_filter(self):
        return {"distance": {"$gt": self.threshold_min,
                             "$lt": self.threshold_max},
                "type": "similar_obj_search",
                "sub_type": {"$in": [x.value for x in self.subtypes]},
                "is_wrong": {"$ne": True}
                }

    def get_pipline_filter(self):
        return {"relations.distance": {"$gt": self.threshold_min, "$lt": self.threshold_max},
                "relations.type": "similar_obj_search",
                "relations.sub_type": {"$in": [x.value for x in self.subtypes]},
                }

    def get_pipline(self):
        tpipline = [
            # Step 1: Filter file records by given record path
            {
                "$match": {
                    "local_path": {"$regex": f"^{re.escape(self.working_folder)}"}
                }
            },
            # Step 2: Attach related detection records by parent_image_id
            {
                "$lookup": {
                    "from": "collection_records",  # Assuming the same collection for detections
                    "localField": "_id",
                    "foreignField": "parent_image_id",
                    "as": "detections"
                }
            },
            # filtrate detections empty
            {
                "$match": {
                    "detections": {"$ne": []},
                    "detections.is_wrong": {"$ne": True}
                }
            },
            # lookup relations collection forign is detection id
            {
                "$lookup": {
                    "from": "relation_records",
                    "localField": "detections._id",
                    "foreignField": "from_id",
                    "as": "relations"
                }
            },
            # filtrate relations by distance
            {
                "$match": self.get_pipline_filter()
            },
            # filtrate empty relations
            {
                "$match": {
                    "relations": {"$ne": []}
                }
            },
        ]
        return tpipline

    def is_subtype_in_filter(self, subtype: DetectionSearchRelSubType):
        return subtype in self.subtypes


class MainView(PySide6GlueWidget):
    sel_checkbox: QCheckBox = None

    def __init__(self):
        self.list_view_etalons = None
        self.prop = Image_graph_app_bind()
        Allocator.register(MainView, self)
        self.window_size = None
        """store window size for later reuse of bild child widgets"""
        MessageSystem.Subscribe(QTMessages.MAIN_WINDOW_RESIZED, self, self.on_main_window_resized)
        self.selection_man = SelectionManager()
        self.similar_graph = Similar_gr_system()
        super().__init__()

    def on_main_window_resized(self, size):
        self.window_size = size

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        g_layout = QHBoxLayout()
        with group():
            threshold_min = wb(
                QLineEdit()).add_to_layout_with(g_layout, [QLabel("Threshold min")]).build()
            threshold_min.textChanged.connect(lambda x: setattr(self.prop, "threshold_min", float(x)))
            threshold_max = QLineEdit()
            threshold_max.textChanged.connect(lambda x: setattr(self.prop, "threshold_max", float(x)))
            g_layout.addWidget(QLabel("Threshold max"))
            g_layout.addWidget(threshold_max)
            work_path_edit = (wb(
                QLineEdit()).set_text(self.similar_graph.rel_filter.working_folder)
                              .add_to_layout_with(g_layout, [QLabel("Working folder")]).build())
            work_path_edit.textChanged.connect(lambda x: self.similar_graph.set_working_folder(x))
            button = QPushButton("Find")
            button.clicked.connect(self.build_list_view)
            g_layout.addWidget(button)
        V_layout.addLayout(g_layout)
        self.list_view_etalons = ListViewWidget()
        self.list_view_etalons.data_list_cursor.items_per_page = 30
        self.list_view_etalons.data_list_cursor.sort_alg["by_path"] = self.sort_by_path
        self.list_view_etalons.data_list_cursor.sort = "by_path"
        self.list_view_etalons.template.itemTemplateSelector.add_template(Detection, lwItemTemplate)
        self.setLayout(V_layout)

        V_layout.addWidget(self.list_view_etalons)

        # references list
        #self.list_view_references = ListViewWidget()
        #self.list_view_references.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        #V_layout.addWidget(self.list_view_references)

    def sort_by_path(self, _list):
        return sorted(_list, key=lambda x: x.parent_file.local_path)

    def build_list_view(self, *args, **kwargs):
        self.list_view_etalons.data_list.clear()
        self.selection_man.selectionUsers.clear()
        self.similar_graph.set_threshold(self.prop.threshold_min, self.prop.threshold_max)
        show_detections = self.similar_graph.get_detections()
        self.list_view_etalons.data_list.extend(show_detections)


class ImageSearchRelHelper:
    @staticmethod
    def get_face_relations():
        """Получает все записи отношений по типу 'similar_obj_search'."""
        return list(RelationRecord.find({"type": "similar_obj_search"}))

    @staticmethod
    def build_detection_list(face_relations):
        """Создает список уникальных обнаружений на основе записей отношений."""
        detections = set()
        for rel in tqdm(face_relations, desc="Processing relations"):
            detections.add(Detection(rel.from_id))
            detections.add(Detection(rel.to_id))
        return list(detections)

    @staticmethod
    def remove_relation(relation_id):
        """Удаляет указанное отношение по его ID."""
        result = RelationRecord.delete_one({"_id": relation_id})
        return result.deleted_count > 0

    @staticmethod
    def update_relation_subtype(relation_id, new_subtype):
        """Обновляет подтип отношения."""
        return RelationRecord.update_one({"_id": relation_id}, {"$set": {"sub_type": new_subtype}})

    @staticmethod
    def filter_relations_by_distance(threshold_min, threshold_max):
        """Фильтрует отношения по диапазону дистанции."""
        return list(RelationRecord.find({
            "distance": {"$gte": threshold_min, "$lte": threshold_max},
            "type": "similar_obj_search"
        }))

    @staticmethod
    def get_recognized_objects_for_class(detection_class):
        """Получает распознанные объекты для данного класса."""
        return list(Recognized_object.find({"obj_class_id": detection_class._id}))

    @staticmethod
    def is_valid_relation(relation):
        """Проверяет корректность записи отношения."""
        return relation and relation.from_id and relation.to_id

    @staticmethod
    def optimize_face_relations(face_relations):
        """
        Оптимизирует связи лиц, удаляя дубликаты и бессмысленные связи.
        Например, удаление циклических и несущественных связей.
        """
        optimized_relations = []
        seen_pairs = set()

        for rel in face_relations:
            rel_pair = tuple(sorted([rel.from_id, rel.to_id]))

            if rel_pair not in seen_pairs:
                seen_pairs.add(rel_pair)
                optimized_relations.append(rel)

        return optimized_relations

    @staticmethod
    def set_detection_as_wrong(detection_id):
        """Помечает обнаружение как 'неверное'."""
        return Detection.update_one({"_id": detection_id}, {"$set": {"is_wrong": True}})


class lwrelItemTemplate(ListViewItemWidget, SelectionManagerUser):
    parent_file: FileRecord = None

    def dispose(self):
        FileRecord.onDelete -= self.on_file_deleted

    def subscribe(self):
        manager = Allocator.get_instance(MainView).selection_man
        self.parent_manager = manager
        manager.register_user(self)
        FileRecord.onDelete += self.on_file_deleted

    def on_file_deleted(self, file):
        if self.parent_file is not None:
            if file._id == self.parent_file._id:
                self.dispose()
                self.parent_list_view.data_list.remove(self.data_context)

    def build_header(self):
        self.subscribe()

        file_rel: RelationRecord = self.data_context
        #1
        rel_vi_type = file_rel.vi_type
        if rel_vi_type == "out":
            target = Detection(file_rel.to_id)
            in_detection = Detection(file_rel.from_id)
        else:
            target = Detection(file_rel.from_id)
            in_detection = Detection(file_rel.to_id)
        self.selection_data.selection = target

        tools_panel = FlowLayout()
        with (group()):  # tools panel
            self.content.addLayout(tools_panel)
            self.sel_checkbox = wb(QCheckBox()).add_to_layout(tools_panel).build()
            self.sel_checkbox.stateChanged.connect(lambda x: self.on_set_selected(x))

            rem_rel_button = (wb(QPushButton("X rel"))
                              .set_fixed_size(25, 25).set_tooltip("Remove relation")
                              .add_to_layout(tools_panel)).build()
            rem_rel_button.clicked.connect(lambda: self.removeRelation())

            wrong_face_button = (wb(QPushButton("X w"))
                                 .set_fixed_size(25, 25).set_tooltip("Mark as wrong")
                                 .add_to_layout(tools_panel)
                                 .build())
            wrong_face_button.clicked.connect(lambda: self.mark_as_wrong())

            tool_button = (wb(QToolButton()).QToolButton_set_compact().set_fixed_size(25, 25).
                           add_to_layout(tools_panel).build())
            with group():
                # add menu to tool button
                menu = QMenu(tool_button)
                wb.add_menu_item(menu, "Open in explorer",
                                 lambda: os.system(f'explorer /select,"{target.parent_file.full_path}"'))
                menu.addAction("Delete from disk", lambda: self.delete_from_disk())
                submenu = menu.addMenu("Relations")
                for subtype in DetectionSearchRelSubType:
                    submenu.addAction(subtype.value, lambda x=subtype.value: self.set_item_sub_type(x))
                # menu.addAction("Move all parent to folder", lambda: self.move_to_folder())
                tool_button.setMenu(menu)

        name_label = QLabel(str(target.parent_file.name))
        self_full_path = target.parent_file.full_path
        in_full_path = in_detection.parent_file.full_path
        if self_full_path == in_full_path:
            #make name label red
            name_label.setStyleSheet("color: red")

        # wrap text
        name_label.setWordWrap(True)
        self.content.addWidget(name_label)

        rel_subtype = file_rel.get_field_val("sub_type")
        rel_subtype_label = QLabel(str(rel_subtype))
        self.rel_subtype_label = rel_subtype_label
        self.content.addWidget(rel_subtype_label)
        self.parent_file = target.parent_file

        detection_path = target.obj_image_path
        dist = file_rel.get_field_val("distance")
        emb_type = file_rel.get_field_val("emb_type")
        detection_pip = target.backend
        hint_text = (
            f"Rel type: {rel_vi_type}\n"
            f"Distance: {dist}\n"
            f"Emb type: {emb_type}\n"
            f"Detect pip: {detection_pip}\n"
        )
        images_layout = QHBoxLayout()
        self.content.addLayout(images_layout)
        with group():
            (wb(QLabel())  # detection image
             .QLabel_set_image(detection_path, 128, 128).
             set_tooltip(hint_text).add_to_layout(images_layout))

            thumbnail_path = self.parent_file.get_thumb("medium")
            (wb(QLabel())  # parent file thumbnail
             .QLabel_set_image(thumbnail_path, 256, 256)
             .add_to_layout(images_layout))

    def set_item_sub_type(self, subtype):
        self.data_context.set_field_val("sub_type", subtype)
        self.rel_subtype_label.setText(subtype)

    def delete_from_disk(self):
        target = Detection(self.data_context.to_id)
        target.parent_file.delete()

    def get_group(self, group_param):
        directory = self.parent_file.local_path
        return directory

    def on_set_selected(self, state):
        if self.sel_checkbox.checkState() == Qt.Checked:
            self.set_selected(True)
            self.parent_list_view.select_item(self)
        else:
            self.set_selected(False)
            self.parent_list_view.deselect_item(self)

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
        rel_filter = Allocator.get_instance(MainView).similar_graph.rel_filter
        parent_lv = self.parent().parent_list

        file_rel: RelationRecord = self.data_context
        file_rel.set_field_val("sub_type", DetectionSearchRelSubType.wrong.value)
        if not rel_filter.is_subtype_in_filter(DetectionSearchRelSubType.wrong):
            parent_lv.data_list.remove(self.data_context)

    def Rem_rel_of(self, detection):
        get_rels = RelationRecord.find({"$or": [{"from_id": detection._id}, {"to_id": detection._id}]})
        parent_lv = self.parent().parent_list

        for rel in get_rels:
            parent_lv.data_list.remove(rel)
            rel.delete_rec()

    def mark_as_wrong(self):
        vur_detection = self.selection_data.selection
        vur_detection.is_wrong = True
        self.Rem_rel_of(vur_detection)
        MessageSystem.SendMessage("detection_marked_as_wrong", vur_detection)


class lwItemTemplate(ListViewItemWidget, SelectionManagerUser):
    recognized_name_label: QLabel = None

    def dispose(self):
        #self.parent_manager.unregister_user(self)
        #self.parent_manager = None
        MessageSystem.Unsubscribe(QTMessages.MAIN_WINDOW_RESIZED, self)
        MessageSystem.Unsubscribe("detection_marked_as_wrong", self)
        MessageSystem.Unsubscribe("detection_type_change", self)
        FileRecord.onDelete -= self.on_file_deleted

    def subscribe(self):
        manager = Allocator.get_instance(MainView).selection_man
        self.parent_manager = manager
        manager.register_user(self)
        MessageSystem.Subscribe(QTMessages.MAIN_WINDOW_RESIZED, self, self.on_main_window_resized)
        MessageSystem.Subscribe("detection_marked_as_wrong", self, self.on_detection_marked_as_wrong)
        MessageSystem.Subscribe("detection_type_change", self, self.on_detection_type_change)
        FileRecord.onDelete += self.on_file_deleted

    def on_file_deleted(self, file):
        if self.selection_data.selection.parent_file._id == file._id:
            self.dispose()
            self.parent_list_view.data_list.remove(self.data_context)

    def build_header(self):
        self.subscribe()
        detection: Detection = self.data_context
        self.selection_data.selection = detection
        self.horiz_layout = QHBoxLayout()
        self.content.addLayout(self.horiz_layout)
        self.vert1 = QVBoxLayout()
        self.vert2 = QVBoxLayout()

        self.horiz_layout.addLayout(self.vert1)
        self.horiz_layout.addLayout(self.vert2)
        name_label = QLabel(str(detection.parent_file.name))
        # wrap text
        name_label.setWordWrap(True)

        self.vert1.addWidget(name_label)
        path = detection.parent_file.local_path
        path_label = QLabel(str(path))
        path_label.setWordWrap(True)
        self.vert1.addWidget(path_label)
        detection_recognized_object = detection.parent_obj
        if detection_recognized_object is not None:
            rec_obj_name = detection_recognized_object.name
        else:
            rec_obj_name = "None"
        self.recognized_name_label = wb(QLabel(rec_obj_name)).add_to_layout(self.vert1).build()

        with group():
            tools_panel = FlowLayout()

            self.sel_checkbox = QCheckBox()
            self.sel_checkbox.stateChanged.connect(lambda x: self.on_set_selected(x))
            tools_panel.addWidget(self.sel_checkbox)

            rem_rel_button = QPushButton("X rel")
            rem_rel_button.setFixedSize(25, 25)
            rem_rel_button.setStyleSheet("font-size: 10px;")
            rem_rel_button.setToolTip("Remove all relations")
            rem_rel_button.clicked.connect(lambda: self.set_wrong_not_selected())
            tools_panel.addWidget(rem_rel_button)

            wrong_face_button = QPushButton("X w")
            wrong_face_button.setFixedSize(25, 25)
            wrong_face_button.setStyleSheet("font-size: 10px;")
            wrong_face_button.setToolTip("Mark as wrong")
            wrong_face_button.clicked.connect(lambda: self.mark_as_wrong())
            tools_panel.addWidget(wrong_face_button)

            tool_button = QToolButton()
            tool_button.setFixedSize(25, 25)
            tool_button.setText("..")
            tool_button.setPopupMode(QToolButton.MenuButtonPopup)
            tools_panel.addWidget(tool_button)
            # add menu to tool button
            menu = QMenu(tool_button)
            menu.addAction("Open in explorer",
                           lambda: os.system(f'explorer /select,"{detection.parent_file.full_path}"'))
            menu.addAction("Edit detection", lambda: self.edit_detection())
            menu.addAction("Edit region", lambda: self.edit_region())
            menu.addAction("Child class equal")
            menu.addAction("Delete from disk", lambda: self.delete_from_disk())
            menu.addAction("move to folder of parent", lambda: self.move_to_parent_folder())
            menu.addAction("Load next rel level", lambda: self.load_next_rel_level())
            submenu = menu.addMenu("Relations")
            for subtype in DetectionSearchRelSubType:
                t_button = QToolButton()
                t_button.setText(subtype.value)
                t_button.clicked.connect(lambda a, x=subtype.value: self.set_items_sub_type(x))
                tools_panel.addWidget(t_button)
                submenu.addAction(subtype.value, lambda x=subtype.value: self.set_items_sub_type(x))
            # menu.addAction("Copy path", lambda: os.system(f'echo {file.full_path} | clip'))
            # menu.addAction("Move all parent to folder", lambda: self.move_to_folder())

            tool_button.setMenu(menu)
        thumbnail_path = detection.obj_image_path
        self.face_label = QLabel()
        try:
            pil_image = PILPool.get_pil_image(thumbnail_path)
            pil_image.thumbnail((256, 256))
            # add thumbnail to widget
            pixmap = pil_to_pixmap(pil_image)
            self.face_label.setPixmap(pixmap)
        except Exception as e:
            self.face_label.setText(str(e))
        self.vert1.addWidget(self.face_label)
        full_image_path = detection.parent_file.full_path
        full_image_label = QLabel()
        # 1
        detection_pip = detection.backend
        detection_class = detection.object_class

        hint_text = (
            f"Detect pip: {detection_pip}\n"
            f"Detect class: {detection_class}\n"
        )
        full_image_label.setToolTip(hint_text)
        try:
            pil_image = PILPool.get_pil_image(full_image_path)
            pil_image.thumbnail((256, 256))
            # add thumbnail to widget
            pixmap = pil_to_pixmap(pil_image)
            full_image_label.setPixmap(pixmap)
        except Exception as e:
            full_image_label.setText(str(e))
        self.vert1.addWidget(full_image_label)

        self.list_view_rel = ListViewWidget()
        self.list_view_rel.template.itemTemplateSelector.add_template(RelationRecord, lwrelItemTemplate)
        self.vert2.addWidget(self.list_view_rel)
        view = Allocator.get_instance(MainView)
        size = view.window_size
        self.list_view_rel.setFixedWidth(size.width() - 320)
        self.vert2.addLayout(tools_panel)
        # get out relations
        rel_filter = copy(view.similar_graph.rel_filter.get_filter())
        rel_filter["from_id"] = detection._id
        r = RelationRecord.find(rel_filter)
        ext_list = []
        self.rel_levels = {0: []}
        for rel in r:
            rel.vi_type = "out"
            in_detection = Detection(rel.from_id)
            if not in_detection.is_wrong:
                ext_list.append(rel)
                self.rel_levels[0].append(rel)

        self.list_view_rel.data_list.extend(ext_list)
        # get in relations
        rel_filter = copy(view.similar_graph.rel_filter.get_filter())
        rel_filter["to_id"] = detection._id
        r = RelationRecord.find(rel_filter)
        ext_list = []
        for rel in r:
            rel.vi_type = "in"
            out_d = Detection(rel.to_id)
            if not out_d.is_wrong:
                ext_list.append(rel)
                self.rel_levels[0].append(rel)
        self.list_view_rel.data_list.extend(ext_list)

    def set_items_sub_type(self, subtype):
        current_rels = self.list_view_rel.data_list_cursor.get_filtered_data()
        for rel in current_rels:
            rel.set_field_val("sub_type", subtype)
        self.list_view_rel.refresh()

    def load_next_rel_level(self):
        levels = self.rel_levels
        current_level = len(levels)
        list_of_detections = self.rel_levels[current_level - 1]
        next_level_list = []
        for rel in list_of_detections:
            if rel.vi_type == "out":
                target = Detection(rel.to_id)
            else:
                target = Detection(rel.from_id)
            rel_filter = copy(Allocator.get_instance(MainView).similar_graph.rel_filter.get_filter())
            rel_filter["from_id"] = target._id
            r = RelationRecord.find(rel_filter)
            ext_list = []
            for rel in r:
                rel.vi_type = "out"
                in_detection = Detection(rel.from_id)
                if not in_detection.is_wrong:
                    ext_list.append(rel)
                    self.rel_levels[current_level].append(rel)

            self.list_view_rel.data_list.extend(ext_list)
            # get in relations
            rel_filter = copy(Allocator.get_instance(MainView).similar_graph.rel_filter.get_filter())
            rel_filter["to_id"] = target._id
            r = RelationRecord.find(rel_filter)
            ext_list = []
            for rel in r:
                rel.vi_type = "in"
                out_d = Detection(rel.to_id)
                if not out_d.is_wrong:
                    ext_list.append(rel)
                    self.rel_levels[current_level].append(rel)
            self.list_view_rel.data_list.extend(ext_list)

    def move_to_parent_folder(self):
        childs = self.list_view_rel.data_list_cursor.get_filtered_data()
        target_folder = self.selection_data.selection.parent_file.local_path
        for rel in childs:
            rel: RelationRecord
            if rel.vi_type == "out":
                detection = Detection(rel.to_id)
            else:
                detection = Detection(rel.from_id)
            detection.parent_file.move_to_folder(target_folder)
        parent_list = self.parent_list_view
        parent_list.refresh()

    def delete_from_disk(self):
        detection = self.selection_data.selection
        detection.parent_file.delete()

    def set_wrong_not_selected(self):
        all_child_rels = self.list_view_rel.data_list_cursor.get_filtered_data()

        selected_rels = self.list_view_rel.get_selected_items()
        for rel in all_child_rels:
            if rel not in selected_rels:
                self.mark_child_rel_as_wrong(rel)
        # clear selection
        self.list_view_rel.clear_selection()

    def mark_child_rel_as_wrong(self, rel):
        rel.set_field_val("sub_type", DetectionSearchRelSubType.wrong.value)
        self.list_view_rel.data_list.remove(rel)

    def edit_region(self):
        from SLM.pySide6Ext.widgets.detection_editor import DetectionEditorDialog
        image_record = self.selection_data.selection.parent_file
        image_detections = Detection.find({"parent_image_id": image_record._id})
        detections_list = []
        image_path = image_record.full_path
        for det in image_detections:
            par_obj_name = "None"
            if det.parent_obj is not None:
                par_obj_name = det.parent_obj.name
            detections_list.append(
                {"id": det._id,
                 "label": par_obj_name,
                 "rect": tuple(det.get_rect("abs_xywh"))
                 })

        completions = ['none', 'wrong']
        face_class = DetectionObjectClass.get("face")
        obj_names = [x.name for x in face_class.get_recognized_objects()]
        completions.extend(obj_names)
        dialog = DetectionEditorDialog(image_path, detections_list, completions,
                                       PySide6GlueApp.current_app._main_window)

        dialog.exec()
        if dialog.result() == 1:
            detections = dialog.get_detections()
            for res_det in detections:
                detection = Detection.find_one({"_id": res_det["id"]})
                detection.set_rect(res_det["rect"], "abs_xywh", edit_name=True)
            self.update_face_thumbnail()

    def update_face_thumbnail(self):
        detection = self.selection_data.selection
        detection.invalidate_cache()
        thumbnail_path = detection.obj_image_path
        try:
            pil_image = PILPool.get_pil_image(thumbnail_path)
            pil_image.thumbnail((256, 256))
            # add thumbnail to widget
            pixmap = pil_to_pixmap(pil_image)
            self.face_label.setPixmap(pixmap)
        except Exception as e:
            self.face_label.setText(str(e))

    def edit_detection(self):
        dialog = EditRecognizedObjectsDialog("Edit detection", PySide6GlueApp.current_app._main_window)
        dialog.exec()
        if dialog.dialog_result:
            new_class = dialog.current_class
            rec_obj = dialog.current_recognized_object
            self.set_recognized(new_class, rec_obj)

    def set_recognized(self, nclass, nrec):
        detection = self.selection_data.selection
        detection.set_class(nclass)
        detection.set_recognized_object(nrec)
        MessageSystem.SendMessage("detection_type_change", detection)
        self.parent_list_view.refresh()  #todo: not optimal implement more optimal

    def on_detection_type_change(self, detection):
        if detection is self.selection_data.selection:
            rec_obj = detection.parent_obj
            if rec_obj is not None:
                self.recognized_name_label.setText(rec_obj.name)
                current_rell = self.list_view_rel.data_list_cursor.get_filtered_data()
                for rel in current_rell:
                    rel: RelationRecord
                    if rel.vi_type == "out":
                        rel_rec_obj = Detection(rel.to_id).parent_obj
                    else:
                        rel_rec_obj = Detection(rel.from_id).parent_obj
                    if rel_rec_obj is None or rec_obj == rel_rec_obj:
                        pass
                    else:
                        rel.set_field_val("is_wrong", True)
                        self.list_view_rel.data_list.remove(rel)

    def on_detection_marked_as_wrong(self, detection):
        child_detections = self.list_view_rel.data_list_cursor.get_filtered_data()
        for rel in child_detections:
            rel: RelationRecord
            if rel.from_id == detection._id or rel.to_id == detection._id:
                self.list_view_rel.data_list.remove(rel)

    def on_set_selected(self, state):
        if self.sel_checkbox.checkState() == Qt.Checked:
            self.set_selected(True)
            self.parent_list_view.select_item(self)
        else:
            self.set_selected(False)
            self.parent_list_view.deselect_item(self)

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

    def mark_as_wrong(self):
        detection = self.selection_data.selection
        detection.is_wrong = True
        MessageSystem.SendMessage("detection_marked_as_wrong", detection)
        self.deleteRelations()

    def deleteRelations(self):
        rl = self.list_view_rel.data_list_cursor.get_filtered_data()
        for r in rl:
            r: RelationRecord
            r.delete_rec()
        self.list_view_rel.data_list.clear()


config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"

QtApp = Similar_graph_app()
QtApp.set_main_widget(MainView())

QtApp.run()
