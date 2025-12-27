import PySide6.QtCore
from PySide6.QtWidgets import QVBoxLayout, QTabWidget, QWidget, QHBoxLayout

from SLM.appGlue.core import Allocator
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord
from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget, PySide6GlueApp
from SLM.pySide6Ext.widgets.tagautocomplete import TagBox
from applications.image_graph_qt.qtExt.image_viewer_widjet import ImageView


class ImageEditorApp(PySide6GlueApp):
    pass


class ImageAttributeEditView(PySide6GlueWidget):
    image_view: ImageView
    tags_widget: TagBox

    def __init__(self):
        Allocator.res.register(self)
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def define_gui(self):
        self.setWindowTitle("Image Attribute Editor")
        self.setGeometry(100, 100, 800, 600)

        # Define your GUI elements here
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        horizontal_layout = QHBoxLayout()
        main_layout.addLayout(horizontal_layout)
        panel1_layout = QVBoxLayout()
        # image canvas layout

        horizontal_layout.addLayout(panel1_layout)
        image_view = ImageView()
        image_view.setMinimumSize(800, 600)
        self.image_view = image_view
        panel1_layout.addWidget(image_view)
        panel2_layout = QVBoxLayout()
        # image adgast layout
        horizontal_layout.addLayout(panel2_layout)
        #tabet wiz tools - propertyes,tags,detection
        tab_widget = QTabWidget()
        panel2_layout.addWidget(tab_widget)
        tags_tab = QWidget()
        tags_tab_layout = QVBoxLayout()
        tags_tab.setLayout(tags_tab_layout)
        tab_widget.addTab(tags_tab, "Tags")
        tags_widget = TagBox(self)
        all_tags = TagRecord.find({})
        tags_str = [tag.fullName for tag in all_tags]
        tags_widget.set_completer(tags_str)
        tags_widget.setMinimumHeight(300)
        tags_widget.setMinimumWidth(300)
        tags_tab_layout.addWidget(tags_widget)
        self.tags_widget = tags_widget
        self.tags_widget.tag_changed_event.connect(self.on_tag_changed)
        image_path = r"E:\rawimagedb\repository\safe repo\asorted images\4\B6Pdifu.jpg"
        # Load the image
        self.load_image(image_path)

    @PySide6.QtCore.Slot(str, object)
    def on_tag_changed(self, str,obj):
        pass

    def load_image(self, image_path):
        file_record = FileRecord.get_record_by_path(image_path)
        self.image_view.load_image(image_path)
        self.image_view.fit_in_view()
        file_tags = TagRecord.get_tags_of_file(file_record)
        string_tags = [tag.fullName for tag in file_tags]
        self.tags_widget.edittags = string_tags
        self.tags_widget.UpdateUI()

if __name__ == "__main__":

    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"

    QtApp = ImageEditorApp()
    QtApp.set_main_widget(ImageAttributeEditView())

    QtApp.run()
