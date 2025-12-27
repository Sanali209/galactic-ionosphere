import json

import re

import loguru
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QFontMetrics, QDrag
from bson import ObjectId

from SLM.appGlue.core import Allocator, Resource, GlueApp

from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord

from SLM.mongoext.mongoQueryTextParser2 import parse_mongo_query_with_normalization
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QMenu, QApplication, QWidget

from SLM.pySide6Ext.dialogsEx import show_string_editor

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget
from SLM.pySide6Ext.binding import binding_load
from SLM.pySide6Ext.widget_dsl.parser import ElementParser
from SLM.pySide6Ext.widgets.ImageWidget import ImageOverlayWidget
from SLM.pySide6Ext.widgets.tools import WidgetBuilder
from applications.image_graph_qt.image_editor import ImageAttributeEditView
from applications.image_graph_qt.mongo_query_helper import MongoQueryHelper
from applications.image_graph_qt.qtExt.components import MongoQueryEditor

binding_load()


class File_find_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        WidgetBuilder.add_menu_item(menu, "add", lambda x: print("add"))
        WidgetBuilder.add_menu_item(menu, "add tag to sel", lambda x: self.on_add_tag())

    def on_add_tag(self):
        string, result = show_string_editor("Edit tag", "")
        if result:
            file_search_view: FileSearchView = Allocator.res.get_by_type_one(FileSearchView)
            file_records = file_search_view.list_view_W.get_selected_items()
            for file_record in file_records:
                tag: TagRecord = TagRecord.get_or_create(fullName=string)
                tag.add_to_file_rec(file_record)


class FileRecordTemplate(ListViewItemWidget):
    name_label: QLabel
    dir_path: QLabel
    image_w: ImageOverlayWidget

    def build_header(self):
        self.content.setContentsMargins(5, 5, 5, 5)
        v_layout = QVBoxLayout()
        self.content.addLayout(v_layout)
        fileRecord: FileRecord = self.data_context

        self.name_label = QLabel(str(fileRecord.name))
        self.name_label.setWordWrap(True)
        v_layout.addWidget(self.name_label)
        self.dir_path = QLabel(str(fileRecord.local_path))
        self.dir_path.setWordWrap(True)
        v_layout.addWidget(self.dir_path)
        self.image_w = ImageOverlayWidget(parent=self)
        v_layout.addWidget(self.image_w)
        self.load_data()

    def load_data(self):
        fileRecord: FileRecord = self.data_context
        thumb_path = fileRecord.get_thumb("medium")
        self.image_w.load_image(thumb_path)
        fileRecord: FileRecord = self.data_context
        self.name_label.setText(str(fileRecord.name))
        self.dir_path.setText(str(fileRecord.local_path))
        text_label = QLabel("Rating: ★★★★☆\nSize: ???\nTags: ???", self.image_w)
        text_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 128);"
            "color: white;"
            "padding: 5px;"
            "font-size: 12px;"
        )
        self.image_w.overlay_layout.addWidget(text_label)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        # Add actions to the context menu
        show_in_view = context_menu.addAction("Show in view")
        edit_action = context_menu.addAction("Edit")
        delete_action = context_menu.addAction("Delete")

        # Connect actions to methods
        show_in_view.triggered.connect(self.on_show_in_view)
        edit_action.triggered.connect(self.on_edit)
        delete_action.triggered.connect(self.on_delete)

        # Execute the menu at the mouse position
        context_menu.exec_(event.globalPos())
        event.accept()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def on_edit(self):
        # Implement edit functionality
        print("Edit tag:", self.data_context)

    def on_delete(self):
        # Implement delete functionality
        print("Delete tag:", self.data_context)

    def on_show_in_view(self):
        file_record: FileRecord = self.data_context

        app = GlueApp.current_app
        app.show_as_separate_window(image_view)
        image_view.load_image(file_record.full_path)


class FileSearchView(PySide6GlueWidget, Resource):
    list_view_W: ListViewWidget
    text_Query_LE: MongoQueryEditor

    def __init__(self):
        Allocator.res.register(self)
        super().__init__()
        self.query_helper = MongoQueryHelper()
        self.setAcceptDrops(True)
        self.start_drag_pos = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        text = event.mimeData().text()
        try:
            json_data = json.loads(text)
            if json_data["type"] == "TagRecord":
                tag_record = TagRecord.find_one({"_id": ObjectId(json_data["id"])})
                if tag_record:
                    # set find text
                    self.text_Query_LE.setText(f'tags REGEX "{tag_record.fullName}"')
        except Exception as e:
            loguru.logger.error(f"error parsing drop event {e}")

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.start_drag_pos:
            # Проверяем, что курсор сместился достаточно далеко для перетаскивания
            if (event.pos() - self.start_drag_pos).manhattanLength() >= QApplication.startDragDistance():
                self.start_drag_pos = None  # Сброс, чтобы избежать дублирования
                self.start_drag()  # Запуск перетаскивания

    def start_drag(self):
        selected_files = self.list_view_W.get_selected_items()
        if not selected_files:
            return  # Если ничего не выбрано — выходим

        mime_data = QMimeData()
        json_data = [{"type": "FileRecord", "id": str(file_record._id)} for file_record in selected_files]
        mime_data.setText(json.dumps(json_data))

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.CopyAction)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_drag_pos = event.pos()

    def define_gui(self):
        self.setWindowTitle("File Search")
        self.setGeometry(0, 0, 1920, 1080)
        self.text_Query_LE: MongoQueryEditor = MongoQueryEditor()
        self.list_view_W = ListViewWidget()
        parser = ElementParser()

        widgets_data = {
            "type": QWidget,
            "instance": self,

            "layout": {
                QVBoxLayout: [
                    {
                        "type": QWidget, 'FixedHeight': 35,
                        "layout": {
                            QHBoxLayout: [
                                {
                                    "type": QWidget,
                                    "instance": self.text_Query_LE,
                                    "key": "text_Query_LE"
                                },
                                {
                                    "type": QPushButton,
                                    "text": "Find",
                                    "on_click": lambda: self.find()
                                },
                            ],
                            "setContentsMargins": (0, 0, 0, 0),
                        }
                    }, {
                        "type": ListViewWidget,
                        "instance": self.list_view_W,
                        "key": "list_view_W",
                    }

                ],
                "setContentsMargins": (0, 0, 0, 0),
            }
        }
        parser.parse(widgets_data)

        font_metrics = QFontMetrics(self.text_Query_LE.font())

        # Calculate height for one line
        line_height = font_metrics.lineSpacing()

        # Set fixed height
        self.text_Query_LE.setFixedHeight(line_height + 4)
        self.text_Query_LE.returnPressed.connect(self.find)

        self.list_view_W.template.itemTemplateSelector.add_template(FileRecord, FileRecordTemplate)
        self.list_view_W.data_list_cursor.sort_alg["by_path"] = lambda _list: sorted(_list, key=lambda x: x.full_path)

    def define_gui2(self):
        self.setWindowTitle("File Search")
        self.setGeometry(100, 100, 800, 600)
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        g_layout = QHBoxLayout()

        self.text_Query_LE = MongoQueryEditor()
        font_metrics = QFontMetrics(self.text_Query_LE.font())

        # Calculate height for one line
        line_height = font_metrics.lineSpacing()

        # Set fixed height
        self.text_Query_LE.setFixedHeight(line_height + 4)
        g_layout.addWidget(self.text_Query_LE)
        button = QPushButton("Find")
        button.clicked.connect(self.find)
        g_layout.addWidget(button)
        V_layout.addLayout(g_layout)
        self.list_view_W = ListViewWidget()
        self.list_view_W.template.itemTemplateSelector.add_template(FileRecord, FileRecordTemplate)
        self.list_view_W.data_list_cursor.sort_alg["by_path"] = lambda _list: sorted(_list, key=lambda x: x.full_path)
        self.list_view_W.data_list_cursor.sort = "by_path"
        self.setLayout(V_layout)

        V_layout.addWidget(self.list_view_W)

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

    def find2(self):
        text = self.text_Query_LE.toPlainText()
        try:
            text = self.escape(text, "\\", [])
            mongo_filter = parse_mongo_query_with_normalization(text)
            mongo_filter = self.recursive_postprocess_filter(mongo_filter)
        except Exception as e:
            mongo_filter = {"$text": {"$search": text}}
            try:
                res = FileRecord.find(mongo_filter)
            except Exception as e:
                mongo_filter = None
                loguru.logger.error(f"error parsing query {e}")
        if mongo_filter:
            try:
                res = FileRecord.find(mongo_filter)
                self.list_view_W.data_list.clear()
                self.list_view_W.data_list.extend(res)
                self.list_view_W.list_update_metric()
            except Exception as e:
                loguru.logger.error(f"error finding files {e}")

    def find(self):
        text = self.text_Query_LE.toPlainText()
        mongo_filter = self.query_helper.parse_and_process(text)
        if mongo_filter:
            try:
                res = FileRecord.find(mongo_filter)
                self.list_view_W.data_list.clear()
                self.list_view_W.data_list.extend(res)
                self.list_view_W.list_update_metric()
            except Exception as e:
                loguru.logger.error(f"error finding files {e}")


if __name__ == "__main__":
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    Allocator.disable_module("VisionModule")

    QtApp = File_find_app()
    image_view = ImageAttributeEditView()
    QtApp.set_main_widget(FileSearchView())

    QtApp.run()
