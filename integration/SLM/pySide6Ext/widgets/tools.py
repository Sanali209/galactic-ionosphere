from PIL import Image
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget, QToolButton

from SLM.destr_worck.bg_worcker import BGTask, BGWorker, TaskQueueSeq
from SLM.files_data_cache.pool import PILPool
from SLM.pySide6Ext.pySide6Q import pil_to_pixmap

BGWorker.instance().add_queue("LoadImageTask", TaskQueueSeq)


class LoadImageTasck(BGTask):
    def __init__(self, image_path, callback=None, width=0, height=0):
        super().__init__()
        self.queue_name = "LoadImageTask"
        self.name = "LoadImageTasck"
        self.image_path = image_path
        self.image = None
        self.on_done(callback)
        self.width = width
        self.height = height

    def task_function(self, *args, **kwargs):
        pimage = PILPool.get_pil_image(self.image_path)
        if self.width > 0 and self.height > 0:
            pimage.thumbnail((self.width, self.height))
        self.image = pil_to_pixmap(pimage)
        yield "done"

    def fire_done(self):
        self.report_progress()
        if self.done_callback is not None:
            self.done_callback(self.image)


class WidgetBuilder:
    def __init__(self, widget: QWidget):
        self.widget: QWidget = widget

    def set_text(self, text):
        if hasattr(self.widget, "setText"):
            self.widget.setText(text)
        return self

    def set_tooltip(self, tooltip):
        self.widget.setToolTip(tooltip)
        return self

    def set_word_wrap(self, word_wrap):
        if hasattr(self.widget, "setWordWrap"):
            self.widget.setWordWrap(word_wrap)
        return self

    def set_fixed_size(self, width, height):
        self.widget.setFixedSize(width, height)
        return self

    def set_stylesheet(self, stylesheet):
        self.widget.setStyleSheet(stylesheet)
        return self

    def set_layout(self, layout):
        if isinstance(self.widget, QWidget):
            self.widget.setLayout(layout)
        return self

    def add_to_layout(self, layout):
        layout.addWidget(self.widget)
        return self

    def add_to_layout_with(self, layout,before_widgets=[],after_widgets=[]):
        for widget in before_widgets:
            layout.addWidget(widget)
        layout.addWidget(self.widget)
        for widget in after_widgets:
            layout.addWidget(widget)
        return self

    def QLabel_set_image(self, image_path, width, height):
        if hasattr(self.widget, "setPixmap"):
            try:
                pil_image = PILPool.get_pil_image(image_path)
                pil_image.thumbnail((width, height))
                pix_map = pil_to_pixmap(pil_image)
                self.widget.setPixmap(pix_map)
            except Exception as e:
                self.widget.setText(str(e))
        return self

    def QLabel_set_image_assinc(self, image_path, width=0, height=0):
        def update_image(pix_map):
            if hasattr(self.widget, "setPixmap"):
                self.widget.setPixmap(pix_map)

        if hasattr(self.widget, "setPixmap"):
            temp_image = Image.new("RGB", (32, 32), (255, 255, 255))
            pix_map = pil_to_pixmap(temp_image)
            self.widget.setPixmap(pix_map)
            BGWorker.instance().add_task(LoadImageTasck(image_path, update_image, width, height))
        return self

    def QToolButton_set_compact(self):
        if hasattr(self.widget, "setPopupMode"):
            self.widget.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.widget.setStyleSheet("""
            QToolButton {
                border: none; /* Убираем границу кнопки */
                padding: 0px; /* Убираем лишние отступы */
            }
            QToolButton::menu-indicator { 
                subcontrol-origin: padding; 
                subcontrol-position: center right; /* Размещаем индикатор справа */
            }
        """)
        return self

    def build(self):
        return self.widget


    @staticmethod
    def add_menu_item(menu, text, action_callback):
        action = QAction(text, menu)
        action.triggered.connect(action_callback)
        menu.addAction(action)
        return menu
