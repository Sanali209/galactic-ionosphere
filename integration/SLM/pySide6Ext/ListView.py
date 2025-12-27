import copy

import PySide6.QtCore
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QWidget, QLayout, QVBoxLayout, QFrame, QHBoxLayout, QLabel, QScrollArea
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLayout, QSizePolicy
from PySide6.QtCore import QSize, QRect, QPoint
from loguru import logger
from tqdm import tqdm

from SLM.appGlue.DAL.DAL import AdapterTemplateSelector
from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo
from SLM.appGlue.DAL.datalist2 import DataListModel, DataViewCursor
from SLM.pySide6Ext.pySide6Q import FlowLayout, clear_layout
import time


# todo implement integration with sel manager

class ListViewItemWidget(QWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.parent_list_view = kwargs.get("list_widget")
        self.data_context = None
        self.main_layout = QVBoxLayout()
        self.selected = False
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setLayout(self.main_layout)
        self.content = QVBoxLayout()
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(0)
        self.main_layout.addLayout(self.content)
        self.setStyleSheet("background-color: white;")
        # Enable mouse events for clicking
        self.setMouseTracking(True)
        self.mose_event_propagate = True

    def build_widget(self, data_context):
        self.data_context = data_context
        self.build_header()

    def build_header(self):
        self.content.addWidget(QPushButton("Item: " + str(self.data_context)))

    def dispose(self):
        pass

    def get_group(self, group_param):
        return "defaulth"

    def mousePressEvent(self, event):

        if self.parent_list_view is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self.parent_list_view.toggle_selection(self)
        if self.mose_event_propagate:
            super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        """Update the widget's visual state based on selection."""
        self.selected = selected
        if self.selected:
            self.setStyleSheet("background-color: lightblue;")  # Change color when selected
        else:
            try:
                self.setStyleSheet("background-color: white;")  # Revert to default color
            except:
                pass


class ListViewGroupWidget(QWidget):
    def __init__(self, data_context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.item_widgets = []
        self.data_context = data_context
        self.main_layout = QVBoxLayout(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.content = QHBoxLayout()
        self.main_layout.addLayout(self.content)
        self.build_header()
        self.items_layout = FlowLayout()
        self.main_layout.addLayout(self.items_layout)
        self.parent_list_view = None

    def build_header(self):
        g_label = QLabel("Group: " + str(self.data_context))
        g_label.setFixedHeight(20)
        self.content.addWidget(QLabel("Group: " + str(self.data_context)))
        pass


class ListViewTemplate:
    def __init__(self):
        self.list_widget = None
        self.itemTemplateSelector = AdapterTemplateSelector(ListViewItemWidget)
        self.groupTemplateSelector = AdapterTemplateSelector(ListViewGroupWidget)


class ListWidgetProps(PropUser):
    current_page: int = PropInfo()
    page_count: int = PropInfo()


class ListViewWidget(QWidget):
    def __init__(self, template: ListViewTemplate = None):
        super().__init__()
        self.prop = ListWidgetProps()
        self.template = template
        if self.template is None:
            self.template = ListViewTemplate()
        self.main_layout = QVBoxLayout(self)

        # Scroll area to hold flow layout
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_container = QWidget()
        self.scroll_container_layout = QVBoxLayout(self.scroll_container)
        self.scroll_container.setLayout(self.scroll_container_layout)
        self.scroll_area.setWidget(self.scroll_container)

        #self.setLayout(self.main_layout)
        self.toolbar = QHBoxLayout()
        self.main_layout.addLayout(self.toolbar)
        self.main_layout.addWidget(self.scroll_area)
        # toolbar tools
        self.next_p_button = QPushButton("Next")
        self.toolbar.addWidget(self.next_p_button)
        self.next_p_button.clicked.connect(self.next_page)
        self.prev_p_button = QPushButton("Prev")
        self.toolbar.addWidget(self.prev_p_button)
        self.prev_p_button.clicked.connect(self.prev_page)
        self.all_page_label = QLabel("0")
        self.toolbar.addWidget(self.all_page_label)
        self.current_page_label = QLabel("0")
        self.toolbar.addWidget(self.current_page_label)
        self.all_items_label = QLabel("0")
        self.toolbar.addWidget(self.all_items_label)

        self.groups_layout = QVBoxLayout()
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        self.groups_layout.setSpacing(0)
        self.scroll_container_layout.addLayout(self.groups_layout)
        self.data_list: DataListModel = DataListModel()
        self.data_list_cursor = DataViewCursor(self.data_list)
        self.data_list_cursor.attach(self)
        self.items_groups = {}
        self.items_nodes = {}
        self.grouping_mode = None
        self.selected_items = []
        self.selection_mode = "single"
        self.sel_message = "selection_changed"
        self.last_selected_item = None
        self.list_changed_callbacks = []
        # condition race update lock
        self.update_lock = PySide6.QtCore.QMutex()

        self.update_signal.connect(self.update)
        self.suspend_update_metric = False

    def list_update_metric(self):
        if self.suspend_update_metric:
            return
        items_all_count = self.data_list_cursor.all_items_count()
        self.all_items_label.setText(f"{items_all_count}")
        self.current_page_label.setText(f"{self.data_list_cursor.current_page}")
        self.all_page_label.setText(f"{self.data_list_cursor.max_page}")

    def next_page(self):

        self.scroll_area.verticalScrollBar().setValue(0)
        self.data_list_cursor.page_next()

    def prev_page(self):

        self.scroll_area.verticalScrollBar().setValue(0)
        # scroll to up
        self.data_list_cursor.page_previous()

    def toggle_selection(self, item_widget: ListViewItemWidget):
        """Handle item click to toggle selection."""
        shift_pressed = QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier
        control_pressed = QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier
        """Handle selection logic."""
        if self.selection_mode == "single" and not shift_pressed and not control_pressed:
            # Deselect all other items
            self.clear_selection()
            self.select_item(item_widget)
        elif self.selection_mode == "multi" or shift_pressed:
            # Toggle selection state for multi-selection mode
            if item_widget.selected:
                self.deselect_item(item_widget)
            else:
                self.select_item(item_widget)
        elif control_pressed:
            # select all in range
            if self.last_selected_item is not None:
                sel_list = [item for item in self.items_nodes.values()]
                try:
                    start = sel_list.index(self.last_selected_item)
                    end = sel_list.index(item_widget)
                    for i in range(start, end):
                        self.select_item(sel_list[i])
                except:
                    pass

        # Trigger selection changed message
        self.fire_list_changed("selection")

    def select_item(self, item_widget: ListViewItemWidget):
        """Select an item and update visual state."""
        item_widget.set_selected(True)
        if item_widget not in self.selected_items:
            self.selected_items.append(item_widget)
        self.last_selected_item = item_widget

    def deselect_item(self, item_widget: ListViewItemWidget):
        """Deselect an item and update visual state."""
        item_widget.set_selected(False)
        if item_widget in self.selected_items:
            self.selected_items.remove(item_widget)

    def get_selected_items(self):
        return [item.data_context for item in self.selected_items]

    def clear_selection(self):
        """Deselect all items."""
        for item in copy.copy(self.selected_items):
            try:
                item.set_selected(False)
            except:
                pass
            if item in self.selected_items:
                self.selected_items.remove(item)

    def get_current_page_data_items(self) -> list:
        ret_list = []
        for item in self.items_nodes.values():
            ret_list.append(item.data_context)
        return ret_list

    def fire_list_changed(self, event):
        for callback in self.list_changed_callbacks:
            callback(event)

    def construct_item(self, item):
        #try:
        template = self.template.itemTemplateSelector.get_template(item)
        lv_item: ListViewItemWidget = template(list_widget=self)
        lv_item.build_widget(item)
        item_group_inst = lv_item.get_group(self.grouping_mode)
        self.items_nodes[item] = lv_item
        # search group if exist
        group_widget = self.items_groups.get(item_group_inst, None)
        if group_widget is None:
            group_widget: ListViewGroupWidget = (self.template.groupTemplateSelector.
                                                 get_template(item_group_inst)(item_group_inst))
            group_widget.parent_list = self
            self.groups_layout.addWidget(group_widget)
            self.items_groups[item_group_inst] = group_widget
        group_widget.items_layout.addWidget(lv_item)
        #except Exception as e:
            #logger.error(f"Error in construct_item: {e}")
        #self.groups_layout.addWidget(lv_item)

    def clear(self):
        clear_layout(self.groups_layout)
        for widget in self.items_nodes.values():
            widget.dispose()
        self.items_groups.clear()
        self.items_nodes.clear()

    clear_signal = PySide6.QtCore.Signal()

    @PySide6.QtCore.Slot()
    def _clear(self):
        pass

    def refresh(self):
        self.data_list_cursor.refresh()

    update_signal = PySide6.QtCore.Signal(object, object, object)

    # implement assinc update
    @PySide6.QtCore.Slot(object, object, object)
    def update(self, data_model, change_type, item=None):
        try:
            if change_type == "refresh":
                self.clear()
                ilist = list(data_model)
                for item in tqdm(ilist, desc="refresh constructing items"):
                    self.construct_item(item)
                self.list_update_metric()
                return

            elif change_type == "add":
                self.construct_item(item)
            elif change_type == "remove":
                item_widget = self.items_nodes.get(item, None)

                if item_widget is not None:
                    item_widget.dispose()
                    item_group_inst = item_widget.get_group(self.grouping_mode)
                    group_widget = self.items_groups.get(item_group_inst, None)
                    if group_widget is not None:
                        group_widget.items_layout.removeWidget(item_widget)
                        item_widget.deleteLater()
                        del self.items_nodes[item]
                        if group_widget.items_layout.count() == 0:
                            group_widget.deleteLater()
                            del self.items_groups[item_group_inst]
                    # add items from page to empty space
                    try:
                        page_items = self.data_list_cursor.get_filtered_data()
                        items_on_data_source = len(page_items)
                        items_on_view = len(self.items_nodes.values())
                        if items_on_data_source > items_on_view:
                            last_item = page_items[-1]
                            if self.items_nodes.get(data_model, None) is None:
                                self.construct_item(last_item)
                    except IndexError:
                        logger.debug("No items on page")
            elif change_type == "clear":
                self.clear()
            self.list_update_metric()
        except Exception as e:
            logger.error(f"Error in update: {e}")

    def list_update(self, data_model, change_type, item=None):
        self.update(data_model, change_type, item)
        #self.update_signal.emit(data_model, change_type, item)
