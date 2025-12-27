import os
import re
from PySide6.QtCore import Qt

from tqdm import tqdm

from SLM.appGlue import core
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.DesignPaterns.specification import Specification, SpecificationBuilder
from SLM.appGlue.core import Allocator
from SLM.appGlue.selectionManager import SelectionManager, SelectionManagerUser
from SLM.destr_worck.bg_worcker import BGWorker
from SLM.files_data_cache.thumbnail import ImageThumbCache
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.object_recognition.object_recognition import Detection
from SLM.groupcontext import group
from SLM.mongoext.mongoQuery import MongoQueryBuilder
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QMenu, QSizePolicy, QToolButton, \
    QCheckBox

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout, pil_to_pixmap, QTMessages, \
    PySide6GlueDockWidget
from SLM.pySide6Ext.binding import binding_load
from SLM.pySide6Ext.widgets.object_editor import ObjectEditorView, PySide6ObjectEditor
from SLM.pySide6Ext.widgets.tools import WidgetBuilder


#todo not delete relation marck as wrong


class Similar_graph_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        # set window size
        WidgetBuilder.add_menu_item(menu, "Sel as wrong", self.mark_as_wrong)
        WidgetBuilder.add_menu_item(menu, "Add Face relations", self.AddFaceRelations)

    def AddFaceRelations(self,e):
        pass

    def mark_as_wrong(self, e):
        selection_manager: SelectionManager = Allocator.get_instance(MainView).selection_man
        sel = selection_manager.get_selection_as()
        for item in sel:
            if item.parent_item is not None:
                item.parent_item.mark_as_wrong()
        selection_manager.clear_selection()


class Image_graph_app_bind(PropUser):
    working_folder = PropInfo(default=r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties")

    def __init__(self):
        super().__init__()


class ImageSearchRelFilter:
    def __init__(self,working_folder=r"E:\rawimagedb\repository"):
        self.working_folder = working_folder

    def get_pipeline(self):
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
                    "detections": {"$ne": []}
                }
            },
        ]
        return tpipline


class RuntimeFilter:
    """
    A filter class used at runtime to apply various filtering criteria.
    criteria show wrong detection
    """
    def __init__(self):
        self.show_wrong = True
        self.specBuilder = SpecificationBuilder()
        self.specBuilder.named_spec={"show_wrong":WrongSpecification}

    def build_filter(self):
        spec = (self.specBuilder.
                add_specification("show_wrong",self.show_wrong).build())
        return spec


class FilterView(PySide6GlueDockWidget):
    def __init__(self):
        self.editor: PySide6ObjectEditor = None
        self.runtime_filter = Allocator.get_instance(MainView).runtime_filter
        Allocator.register(FilterView, self)
        super().__init__()




    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        self.setLayout(V_layout)
        self.editor = PySide6ObjectEditor()
        V_layout.addWidget(self.editor)
        self.editor.add_view_template(RuntimeFilter, RuntimeFilterEditor)
        self.editor.set_object(self.runtime_filter)

class WrongSpecification(Specification):
    def __init__(self,value):
        self.value = value

    def is_satisfied_by(self, candidate):
        candidate: Detection
        if self.value:
            return True
        else:
            return not candidate.is_wrong

class RuntimeFilterEditor(ObjectEditorView):
    def __init__(self, edit_object: object):
        super().__init__(edit_object)
        edit_object: RuntimeFilter
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        WidgetBuilder(QLabel("Filter")).add_to_layout(main_layout)
        self.check_box:QCheckBox = WidgetBuilder(QCheckBox()).set_text("Show wrong").add_to_layout(main_layout).build()
        self.check_box.setChecked(edit_object.show_wrong)
        self.check_box.stateChanged.connect(lambda x: self.edit_is_wrong(x))

        appy_filter_button = WidgetBuilder(QPushButton("Apply filter")).add_to_layout(main_layout).build()
        appy_filter_button.clicked.connect(self.on_apply_filter)

    def edit_is_wrong(self, state):
        if self.check_box.checkState() == Qt.CheckState.Checked:
            self.object.show_wrong = True
        else:
            self.object.show_wrong = False

    def on_apply_filter(self):
        main_view = Allocator.get_instance(MainView)
        list_view:ListViewWidget = main_view.list_view_detections
        list_view.data_list_cursor.set_specification(main_view.runtime_filter.build_filter())
        list_view.suspend_update_metric = True
        list_view.refresh()
        list_view.suspend_update_metric = False
        list_view.list_update_metric()



class MainView(PySide6GlueWidget):

    def __init__(self):
        self.rel_filter = ImageSearchRelFilter()
        self.runtime_filter = RuntimeFilter()
        self.list_view_detections:ListViewWidget = None
        self.prop = Image_graph_app_bind()
        Allocator.register(MainView, self)
        self.window_size = None
        MessageSystem.Subscribe(QTMessages.MAIN_WINDOW_RESIZED, self, self.on_main_window_resized)
        self.selection_man = SelectionManager()

        super().__init__()

    def on_main_window_resized(self, size):
        self.window_size = size

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        g_layout = QHBoxLayout()

        work_path_edit = QLineEdit()
        work_path_edit.setText(self.rel_filter.working_folder)
        work_path_edit.textChanged.connect(lambda x: self.set_folder(x))
        g_layout.addWidget(QLabel("Working folder"))
        g_layout.addWidget(work_path_edit)
        button = QPushButton("Find")
        button.clicked.connect(self.build_list_view)
        g_layout.addWidget(button)
        V_layout.addLayout(g_layout)
        self.list_view_detections = ListViewWidget()
        self.list_view_detections.data_list_cursor.items_per_page = 100
        self.list_view_detections.template.itemTemplateSelector.add_template(Detection, lwItemTemplate)
        self.list_view_detections.data_list_cursor.set_specification(self.runtime_filter.build_filter())
        self.setLayout(V_layout)

        V_layout.addWidget(self.list_view_detections)

        # references list
        #self.list_view_references = ListViewWidget()
        #self.list_view_references.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        #V_layout.addWidget(self.list_view_references)
        parent_app: PySide6GlueApp = core.Allocator.get_instance(PySide6GlueApp)
        parent_app.add_dock_widget(FilterView(), Qt.DockWidgetArea.LeftDockWidgetArea)

    def build_list_view(self, *args, **kwargs):
        self.list_view_detections.data_list.clear()
        self.selection_man.selectionUsers.clear()
        pipeline = self.rel_filter.get_pipeline()
        res = FileRecord.collection().aggregate(pipeline)
        res_list = list(res)
        show_detections = []
        for rec in tqdm(res_list):
            for detection in rec["detections"]:
                show_detections.append(detection["_id"])
        for detection_id in show_detections:
            self.list_view_detections.suspend_update_metric = True
            self.list_view_detections.data_list.append(Detection(detection_id))
        self.list_view_detections.suspend_update_metric = False
        self.list_view_detections.list_update_metric()



    def set_folder(self, folder):
        self.rel_filter.working_folder = folder


class lwItemTemplate(ListViewItemWidget, SelectionManagerUser):

    def build_header(self):
        manager = Allocator.get_instance(MainView).selection_man
        self.parent_manager = manager
        manager.register_user(self)
        detection: Detection = self.data_context
        detection.parent_item = self
        self.selection_data.selection = detection

        name_label = (WidgetBuilder(QLabel(str(detection.parent_file.name)))
                      .set_word_wrap(True)
                      .add_to_layout(self.content)
                      ).build()
        # wrap text
        detection_pip = detection.backend
        detection_pip_label = (WidgetBuilder(QLabel(str(detection_pip)))
                               .add_to_layout(self.content)
                               ).build()
        path = detection.parent_file.local_path
        path_label = (WidgetBuilder(QLabel(str(path)))
                      .set_word_wrap(True)
                      .add_to_layout(self.content)
                      ).build()

        with group():
            tools_panel = FlowLayout()
            self.content.addLayout(tools_panel)

            self.sel_checkbox = WidgetBuilder(QCheckBox()).add_to_layout(tools_panel).set_text("sel").build()
            self.sel_checkbox.stateChanged.connect(lambda x: self.on_set_selected(x))

            self.is_wrong_checkbox = WidgetBuilder(QCheckBox()).add_to_layout(tools_panel).set_text("Wrong").build()
            if detection.is_wrong:
                self.is_wrong_checkbox.setCheckState(Qt.CheckState.Checked)
            self.is_wrong_checkbox.stateChanged.connect(lambda x: self.on_wrong_set(x))

            tool_button = (WidgetBuilder(QToolButton())
                           .set_text("..")
                           .set_fixed_size(25, 25)
                           .add_to_layout(tools_panel)
                           ).build()
            tool_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            # add menu to tool button
            menu = QMenu(tool_button)
            menu.addAction("Open in explorer",
                           lambda: os.system(f'explorer /select,"{detection.parent_file.full_path}"'))
            # menu.addAction("Open in default app", lambda: os.system(f'start "" "{file.full_path}"'))
            # menu.addAction("Copy path", lambda: os.system(f'echo {file.full_path} | clip'))
            # menu.addAction("Move all parent to folder", lambda: self.move_to_folder())

            tool_button.setMenu(menu)
        detection_image_path = detection.obj_image_path
        (WidgetBuilder(QLabel())
         .QLabel_set_image(detection_image_path, 256, 256)
         .add_to_layout(self.content)
         ).build()
        full_image_path = detection.parent_file.full_path
        full_thumb_path = ImageThumbCache.instance().get_thumb(full_image_path)
        (WidgetBuilder(QLabel())
         .QLabel_set_image(full_thumb_path, 256, 256)
         .add_to_layout(self.content)
         ).build()

    def dispose(self):
        #self.parent_manager.unregister_user(self)
        #self.parent_manager = None
        MessageSystem.Unsubscribe(QTMessages.MAIN_WINDOW_RESIZED, self)

    def on_set_selected(self, state):
        if self.sel_checkbox.checkState() == Qt.CheckState.Checked:
            self.set_selected(True)
        else:
            self.set_selected(False)

    def on_wrong_set(self, state):
        if self.is_wrong_checkbox.checkState() == Qt.CheckState.Checked:
            self.is_wrong = True
        else:
            self.is_wrong = False
        detection = self.data_context
        detection.is_wrong = self.is_wrong

    def set_selected(self, selected: bool):
        try:
            if selected:
                self.sel_checkbox.setCheckState(Qt.CheckState.Checked)
            else:
                self.sel_checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.sel_checkbox.update()
        except Exception as e:
            print(e)
        super().set_selected(selected)

    def mark_as_wrong(self):
        detection = self.data_context
        detection.is_wrong = True
        if self.is_wrong_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.is_wrong_checkbox.setCheckState(Qt.CheckState.Checked)
        #self.deleteRelations()

    def deleteRelations(self):
        query_builder = MongoQueryBuilder()
        or_query = query_builder.build_or(
            {"from_id": self.data_context._id},
            {"to_id": self.data_context._id}
        )
        final_query = MongoQueryBuilder() \
            .add_condition("type", "similar_face_search") \
            .add_and(or_query) \
            .build()

        relations = RelationRecord.find(final_query)
        for relation in relations:
            relation.delete()


binding_load()
config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"

QtApp = Similar_graph_app()
QtApp.set_main_widget(MainView())

QtApp.run()

BGWorker.instance().stop()
