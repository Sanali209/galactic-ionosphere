import sys
from enum import Enum

from PySide6.QtGui import QPixmap, QImage, QPainter, QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLayout, QSizePolicy, QDockWidget, QMenu

from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QLayout
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QLayout, \
    QSizePolicy
from PySide6.QtCore import QRect, QSize, QPoint, Qt

from SLM.appGlue.core import GlueApp, Allocator, Resource


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def pil_to_pixmap(pil_image):
    """Convert PIL Image to QPixmap."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    data = pil_image.tobytes("raw", "RGB")
    bytes_per_line = pil_image.width * 3
    qimage = QImage(data, pil_image.width, pil_image.height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, h_spacing=1, v_spacing=1):
        super(FlowLayout, self).__init__(parent)
        self.item_list = []
        self.hspacing = h_spacing
        self.vspacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self.item_list.append(item)

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().left(), 2 * self.contentsMargins().top())
        return size

    def do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0

        for item in self.item_list:
            next_x = x + item.sizeHint().width() + self.hspacing
            if next_x - self.hspacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self.vspacing
                next_x = x + item.sizeHint().width() + self.hspacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

    # This is the key method to ensure the layout works with scroll areas
    def expandingDirections(self):
        return Qt.Orientation(Qt.Horizontal | Qt.Vertical)

    def minimumSize(self):
        size = QSize(0, 0)
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def sizeHint(self):
        return self.minimumSize()


class PySide6GlueWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.paren_app = None
        self.is_secondary_window = False
        self.define_gui()

    def on_window_close(self, e):
        if self.is_secondary_window:
            self.paren_app.nested_widgets.remove(self)
            self.close()

    def define_gui(self):
        pass
class ToolPanelManager:
    def __init__(self, main_window):
        """
        Менеджер панелей инструментов.
        :param main_window: Главный объект окна (CustomQMainWindow).
        """
        self.main_window = main_window
        self.dock_widgets = {}  # Хранение зарегистрированных виджетов: {имя: QDockWidget}

    def add_tool(self, name, widget, area=Qt.LeftDockWidgetArea, closable=True):
        """
        Добавить новый инструмент.
        :param name: Имя инструмента.
        :param widget: Содержимое QDockWidget.
        :param area: Область размещения виджета (Qt.DockWidgetArea).
        :param closable: Можно ли закрыть виджет.
        """
        if name in self.dock_widgets:
            raise ValueError(f"Инструмент с именем {name} уже существует.")

        dock_widget = QDockWidget(name, self.main_window)
        dock_widget.setWidget(widget)
        dock_widget.setFeatures(
            QDockWidget.DockWidgetClosable if closable else QDockWidget.NoDockWidgetFeatures
        )
        self.main_window.addDockWidget(area, dock_widget)
        self.dock_widgets[name] = dock_widget

    def remove_tool(self, name):
        """
        Удалить инструмент.
        :param name: Имя инструмента.
        """
        dock_widget = self.dock_widgets.pop(name, None)
        if dock_widget:
            dock_widget.setParent(None)
            dock_widget.deleteLater()

    def show_tool(self, name):
        """
        Показать инструмент, если он скрыт.
        :param name: Имя инструмента.
        """
        dock_widget = self.dock_widgets.get(name)
        if dock_widget:
            dock_widget.show()

    def hide_tool(self, name):
        """
        Скрыть инструмент.
        :param name: Имя инструмента.
        """
        dock_widget = self.dock_widgets.get(name)
        if dock_widget:
            dock_widget.hide()

    def toggle_tool(self, name):
        """
        Переключить видимость инструмента.
        :param name: Имя инструмента.
        """
        dock_widget = self.dock_widgets.get(name)
        if dock_widget:
            dock_widget.setVisible(not dock_widget.isVisible())

    def create_menu(self, parent_menu):
        """
        Создать меню управления инструментами.
        :param parent_menu: Родительское меню.
        """
        tools_menu = QMenu("Инструменты", parent_menu)
        for name in self.dock_widgets.keys():
            action = QAction(name, tools_menu)
            action.setCheckable(True)
            action.setChecked(self.dock_widgets[name].isVisible())
            action.toggled.connect(lambda checked, n=name: self.show_tool(n) if checked else self.hide_tool(n))
            tools_menu.addAction(action)
        parent_menu.addMenu(tools_menu)

# refactor with clas parented to Q
class PySide6GlueDockWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._dock_widget = QDockWidget()
        self._dock_widget.setWidget(self)  #set widget content
        self.define_gui()

    def define_gui(self):
        pass


class QTMessages(Enum):
    MAIN_WINDOW_RESIZED = "MAIN_WINDOW_RESIZED"


class CustomQMainWindow(QMainWindow):
    def resizeEvent(self, event):
        # Get the new size of the window
        new_size = self.size()
        # send the new size to the message system
        MessageSystem.SendMessage(QTMessages.MAIN_WINDOW_RESIZED, new_size)
        # Call the parent class's resizeEvent to ensure default behavior
        super().resizeEvent(event)


class PySide6GlueApp(GlueApp):
    def __init__(self):
        super().__init__()
        self._qapp = QApplication(sys.argv)
        self._qapp.setStyleSheet("QToolTip { background-color: white; color: black; border: 0px; }")
        self._main_window = CustomQMainWindow()
        self.nested_widgets = []
        self.app_central_widget = None

    def set_main_widget(self, widget):
        widget.paren_app = self
        self.app_central_widget = widget
        self.nested_widgets.append(widget)
        self._main_window.setCentralWidget(widget)

    def show_as_separate_window(self, widget):
        self.nested_widgets.append(widget)
        widget.is_secondary_window = True
        widget.paren_app = self
        widget.closeEvent = widget.on_window_close
        widget.show()

    def show_window_modal(self, widget):
        self.nested_widgets.append(widget)
        widget.is_secondary_window = True
        widget.paren_app = self
        widget.setWindowModality(Qt.ApplicationModal)
        widget.closeEvent = widget.on_window_close
        widget.show()

    def add_dock_widget(self, dock_widget, area):
        self._main_window.addDockWidget(area, dock_widget._dock_widget)

    def run(self):
        super().run()
        self._qapp.setQuitOnLastWindowClosed(True)
        self._main_window.show()
        self._qapp.exec_()

    def exit(self):
        super().exit()
        self._qapp.quit()
