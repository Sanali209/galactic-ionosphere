import json
import os
import re
import subprocess
from collections import defaultdict
from copy import copy
from enum import Enum

from PySide6.QtCore import Qt, QLoggingCategory, QMimeData
from PySide6.QtGui import QAction, QDrag

from tqdm import tqdm
from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.core import Allocator
from SLM.appGlue.selectionManager import SelectionManager, SelectionManagerUser
from SLM.destr_worck.bg_worcker import BGTask
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.groupcontext import group
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QMenu, QToolButton, \
    QCheckBox

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo
from SLM.pySide6Ext.dialogsEx import show_string_editor

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout, QTMessages
from SLM.pySide6Ext.binding import binding_load
from SLM.pySide6Ext.widgets.ImageWidget import ImageOverlayWidget
from SLM.pySide6Ext.widgets.tools import WidgetBuilder

binding_load()

QLoggingCategory.setFilterRules("*.debug=true")

import faulthandler

faulthandler.enable()




class Tag_edit_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        WidgetBuilder.add_menu_item(menu, "add", lambda x: print("add"))
        WidgetBuilder.add_menu_item(menu, "remap exist tags", lambda x: self.on_remap_exist())

    def on_remap_exist(self):
        tag_edit_view:TagEditView = Allocator.get_instance(TagEditView)
        tags = tag_edit_view.list_view_W.data_list_cursor.get_filtered_data()
        for tag in tags:
            tag.remap_tag()

class Image_graph_app_bind(PropUser):
    threshold_min = PropInfo(default=0.0)
    threshold_max = PropInfo(default=1.0)
    working_folder = PropInfo(default=r"E:\rawimagedb\repository\nsfv repo\drawn\presort\buties")
    compare_image_list = PropInfo(default=[])

    def __init__(self):
        super().__init__()
        self.test_text = "Hello World"


class EditTagDialog(PySide6GlueWidget):
    pass


class TagLWTemplate(ListViewItemWidget):
    name_label: QLabel
    full_name: QLabel
    autotag_check: QCheckBox

    def build_header(self):

        self.content.setContentsMargins(5, 5, 5, 5)
        v_layout = QVBoxLayout()
        self.content.addLayout(v_layout)
        self.name_label = QLabel()
        v_layout.addWidget(self.name_label)
        self.full_name = QLabel()
        v_layout.addWidget(self.full_name)
        self.autotag_check = QCheckBox()
        self.autotag_check.stateChanged.connect(self.on_autotag_change)
        v_layout.addWidget(self.autotag_check)
        self.remap_line_edit = QLineEdit()
        self.remap_line_edit.setPlaceholderText("Remap tag")
        self.remap_line_edit.textChanged.connect(self.on_remap_change)
        v_layout.addWidget(self.remap_line_edit)
        self.load_data()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            mime_data = QMimeData()
            json_data = {"type": "TagRecord", "id": str(self.data_context._id)}
            mime_data.setText(json.dumps(json_data))
            drag = QDrag(self)
            drag.setMimeData(mime_data)
            drag.exec_(Qt.CopyAction)

    def load_data(self):
        tag: TagRecord = self.data_context
        self.name_label.setText(tag.name)
        self.full_name.setText(tag.fullName)
        checked: bool = tag.autotag
        if checked is None:
            checked = False
        self.autotag_check.setChecked(checked)
        self.remap_line_edit.setText(tag.remap_to_tags)

    def on_remap_change(self, text):
        tag: TagRecord = self.data_context
        tag.remap_to_tags = text

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        # Add actions to the context menu
        edit_action = context_menu.addAction("Edit")
        delete_action = context_menu.addAction("Delete")
        delete_record_action = context_menu.addAction("Delete record")
        marck_as_tag = context_menu.addAction("Mark as tag")

        # Connect actions to methods
        edit_action.triggered.connect(self.on_edit)
        delete_action.triggered.connect(self.on_delete)
        delete_record_action.triggered.connect(self.on_delete_record)
        marck_as_tag.triggered.connect(self.on_marck_as_tag)

        # Execute the menu at the mouse position
        context_menu.exec_(event.globalPos())

    def on_marck_as_tag(self):
        tag: TagRecord = self.data_context
        tag.remap_tag()

    def on_autotag_change(self, state):
        tag: TagRecord = self.data_context
        tag.autotag = state == Qt.Checked

    def on_edit(self):
        tag: TagRecord = self.data_context
        string, result = show_string_editor("Edit tag", tag.fullName)
        if result:
            tag.rename(string)

            self.load_data()

    def on_delete(self):
        # Implement delete functionality
        tag: TagRecord = self.data_context
        tag.delete()
        self.parent_list_view.data_list.remove(tag)

    def on_delete_record(self):
        # Implement delete functionality
        tag: TagRecord = self.data_context
        tag.delete_rec()
        self.parent_list_view.data_list.remove(tag)
        self.parent_list_view.list_update_metric()


class TagEditView(PySide6GlueWidget):
    list_view_W: ListViewWidget
    text_Query_LE: QLineEdit

    def __init__(self):
        Allocator.register(TagEditView, self)
        super().__init__()

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        g_layout = QHBoxLayout()
        self.text_Query_LE = QLineEdit()
        g_layout.addWidget(self.text_Query_LE)
        button = QPushButton("Find")
        button.clicked.connect(self.find)
        g_layout.addWidget(button)
        V_layout.addLayout(g_layout)
        self.list_view_W = ListViewWidget()
        self.list_view_W.data_list_cursor.sort_alg["by_full_name"] = lambda _list: sorted(_list, key=lambda x: x.fullName)
        self.list_view_W.data_list_cursor.sort = "by_full_name"
        self.list_view_W.template.itemTemplateSelector.add_template(TagRecord, TagLWTemplate)
        self.setLayout(V_layout)

        V_layout.addWidget(self.list_view_W)

    def find(self):
        text = self.text_Query_LE.text()
        res = []
        if text == "":
            res = TagRecord.find({})
        if text:
            res = TagRecord.find({"fullName": {"$regex": text}})
        self.list_view_W.data_list.clear()
        self.list_view_W.data_list.extend(res)
        self.list_view_W.list_update_metric()


config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"

QtApp = Tag_edit_app()
QtApp.set_main_widget(TagEditView())

QtApp.run()
