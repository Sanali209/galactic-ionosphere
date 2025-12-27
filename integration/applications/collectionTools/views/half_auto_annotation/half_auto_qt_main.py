import os
from PySide6.QtCore import Qt
from tqdm import tqdm

from SLM.appGlue.DAL.DAL import GlueDataConverter
from SLM.appGlue.core import Allocator
from SLM.destr_worck.bg_worcker import BGWorker, BGTask
from SLM.files_data_cache.pool import PILPool

os.environ['DATA_CACHE_MANAGER_PATH'] = r'D:\data\ImageDataManager'
from PIL import Image

from SLM.flet.bindings.binds import BaseIterableToStrConverter
from SLM.groupcontext import group

from applications.collectionTools.views.half_auto_annotation.annotation_prediction import AnnotationPredictionManager
from SLM.files_db.annotation_tool.annotation import AnnotationJob, SLMAnnotationClient

from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.pySide6Ext.ListView import ListViewWidget, ListViewItemWidget

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QComboBox, QSizePolicy, \
    QToolButton, QFileDialog, QMenu

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget, FlowLayout, pil_to_pixmap
import SLM.pySide6Ext.binding


class QuickImageViewer(PySide6GlueWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        self.label = QLabel()
        V_layout.addWidget(self.label)
        self.setLayout(V_layout)

    def set_image(self, image_path):
        self.image_path = image_path
        try:
            pil_image = PILPool.get_pil_image(image_path)
            pil_image.thumbnail((800, 800))
            pixmap = pil_to_pixmap(pil_image)
            self.label.setPixmap(pixmap)
        except Exception as e:
            self.label.setText(str(e))


class Half_auto_annotator_app(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        menu_bar = self._main_window.menuBar()
        menu = menu_bar.addMenu("File")
        # set window size
        #self._main_window.resize(800, 600)
        menu.addAction("Add folder to annotation", self.add_folder_to_annotation)
        menu.addAction("Exit")

    def add_folder_to_annotation(self):
        pass


class Annotate_task(BGTask):
    def __init__(self, file_record, job, label):
        super().__init__()
        self.name = "annotate_task" + str(file_record)
        self.exclude_names = [self.name]
        self.file_record = file_record
        self.job = job
        self.label = label

    def task_function(self, *args, **kwargs):
        self.job.annotate_file(self.file_record, self.label, override_annotation=True)
        yield "done"


class AnnotationRecord_FileRecord(GlueDataConverter):
    def Convert(self, data):
        return data.file

    def ConvertBack(self, data):
        return None


class Show_category_task(BGTask):
    def __init__(self):
        super().__init__()
        self.name = "show_category_task"
        self.exclude_names = [self.name]
        self.cancel_names = ["show_anomaly_task", "show_category_task"]
        self.items = []
        self.list_clear = True
        self.label = None
        self.all_items_count = 0
        self.current_item_count = 0

    def task_function(self, *args, **kwargs):
        gr_editor_ins: Half_auto_editor_widget = Allocator.get_instance(Half_auto_editor_widget)
        if self.list_clear:
            gr_editor_ins.list_view.data_list.clear()
            self.all_items_count = len(self.items)
        if self.label is None:
            self.label = gr_editor_ins.job_label_dropdown.currentText()
            if self.label is None:
                return
        annotator = AnnotationPredictionManager.instance().get_annotator_by_name(
            gr_editor_ins.prediction_pipline_dropdown.currentText())
        curent_items = self.items[0:5000]
        next_items = self.items[5000:]
        converter = AnnotationRecord_FileRecord()
        for item in tqdm(curent_items):
            self.current_item_count += 1
            self.state.progress = self.current_item_count / self.all_items_count

            f_item = converter.Convert(item)
            if annotator.is_satisfied_by(self.label, f_item):
                gr_editor_ins.list_view.data_list.append(f_item)
            #time.sleep(0.003)
            yield None
        if len(next_items) > 0:
            n_task = Show_category_task()
            n_task.all_items_count = self.all_items_count
            n_task.current_item_count = self.current_item_count
            n_task.items = next_items
            n_task.list_clear = False
            n_task.label = self.label
            BGWorker.instance().add_task(n_task, ignore_excludes=True)
        yield "done"


class show_anotated_category_task(BGTask):
    def __init__(self):
        super().__init__()
        self.name = "show_anotated"
        self.exclude_names = [self.name]
        self.cancel_names = ["show_anomaly_task", "show_category_task"]

    def task_function(self, *args, **kwargs):
        gr_editor_ins: Half_auto_editor_widget = Allocator.get_instance(Half_auto_editor_widget)
        job = gr_editor_ins.prop.current_job
        label = gr_editor_ins.job_label_dropdown.currentText()
        items = job.get_ann_records_by_label(label)
        converter = AnnotationRecord_FileRecord()
        for item in items:
            f_item = converter.Convert(item)
            gr_editor_ins.list_view.data_list.append(f_item)
        yield "done"


class Half_auto_editor_widget(PySide6GlueWidget):
    class group_editor_bindings(PropUser):
        all_jobs: list[AnnotationJob] = PropInfo()
        current_job: AnnotationJob = PropInfo()
        currentImageRecord: FileRecord = PropInfo()
        current_work_path: str = PropInfo()

        def __init__(self):
            super().__init__()
            annotation_db = SLMAnnotationClient()
            all_jobs = annotation_db.get_all_jobs({'type': "multiclass/image"})
            self.all_jobs = all_jobs
            valg = self.all_jobs
            if len(valg) > 0:
                self.current_job = self.all_jobs[0]

    def __init__(self):
        Allocator.register(Half_auto_editor_widget, self)
        self.stop_show_button = None
        self.show_button: QPushButton = None
        self.prediction_pipline_dropdown: QComboBox = QComboBox()
        self.job_label_dropdown: QComboBox = QComboBox()
        self.active_job_dropdown: QComboBox = None
        self.list_view: ListViewWidget = ListViewWidget()
        self.prop = Half_auto_editor_widget.group_editor_bindings()
        self.annotation_db = SLMAnnotationClient()
        super().__init__()

    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        V_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMaximumSize)
        with group():
            h_layout = QHBoxLayout()
            V_layout.addLayout(h_layout)
            self.active_job_dropdown = QComboBox()
            h_layout.addWidget(self.active_job_dropdown)
            #self.active_job_dropdown.setFixedHeight(40)
            self.active_job_dropdown.addItems([str(job) for job in self.prop.all_jobs])
            conv = BaseIterableToStrConverter(self.prop.all_jobs)
            self.prop.dispatcher.current_job.bind(bind_target=self.active_job_dropdown, field="value", converter=conv,
                                                  callback=self.on_job_selected)

            h_layout.addWidget(self.job_label_dropdown)
            #self.job_label_dropdown.setFixedHeight(40)

            h_layout.addWidget(self.prediction_pipline_dropdown)
            #self.prediction_pipline_dropdown.setFixedHeight(40)

            self.show_button = QPushButton("Show")
            #self.show_button.setFixedHeight(40)
            self.show_button.clicked.connect(self.on_show_button_click)
            h_layout.addWidget(self.show_button)

            self.stop_show_button = QPushButton("Stop")
            #self.stop_show_button.setFixedHeight(40)
            h_layout.addWidget(self.stop_show_button)
            self.stop_show_button.clicked.connect(self.on_stop_show_button_click)

        self.list_view.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        V_layout.addWidget(self.list_view)

        self.setLayout(V_layout)

    def on_show_button_click(self, *args, **kwargs):
        task = Show_category_task()
        job_items = self.prop.current_job.coll_view.get_filtered_data()

        task.items = job_items
        BGWorker.instance().add_task(task)

    def on_stop_show_button_click(self, *args, **kwargs):
        BGWorker.instance().cancel_task_by_names(["show_category_task", "show_anomaly_task"])

    def on_job_selected(self, *args, **kwargs):
        if self.prop.current_job is not None:
            # initialize job label dropdown
            labels = self.prop.current_job.choices
            if isinstance(labels, list) and len(labels) > 0:
                self.job_label_dropdown.clear()
                self.job_label_dropdown.addItems([x for x in labels])
                self.job_label_dropdown.setCurrentText(labels[0])

            # initialize prediction pipline dropdown
            an_pred_manager = AnnotationPredictionManager.instance()
            annotators = an_pred_manager.get_compatible_annotators(self.prop.current_job.name)
            if len(annotators) > 0:
                self.prediction_pipline_dropdown.clear()
                self.prediction_pipline_dropdown.addItems([x for x in annotators])
                self.prediction_pipline_dropdown.value = annotators[0]


class Folder_annotator_widget(PySide6GlueWidget):
    def define_gui(self):
        V_layout = QVBoxLayout()
        V_layout.setContentsMargins(0, 0, 0, 0)
        V_layout.setSpacing(0)
        self.setLayout(V_layout)


class lwItemTemplate(ListViewItemWidget):

    def build_header(self):
        file: FileRecord = self.data_context
        name_label = QLabel(str(file.name))
        # wrap text
        name_label.setWordWrap(True)
        self.content.addWidget(name_label)
        thumbnail_path = file.get_thumb("medium")

        with group():
            horiz_layout = QHBoxLayout()
            self.content.addLayout(horiz_layout)

            annotate_button = QPushButton("Annotate")
            horiz_layout.addWidget(annotate_button)
            annotate_button.clicked.connect(self.on_annotate_button_click)

            tool_button = QToolButton()
            tool_button.setText("..")
            tool_button.setPopupMode(QToolButton.MenuButtonPopup)
            horiz_layout.addWidget(tool_button)
            # add menu to tool button
            menu = QMenu(tool_button)
            menu.addAction("Open in explorer", lambda: os.system(f'explorer /select,"{file.full_path}"'))
            menu.addAction("Open in default app", lambda: os.system(f'start "" "{file.full_path}"'))
            menu.addAction("Copy path", lambda: os.system(f'echo {file.full_path} | clip'))
            menu.addAction("Move all parent to folder", lambda: self.move_to_folder())

            tool_button.setMenu(menu)

        label = QLabel()
        try:
            pil_image = PILPool.get_pil_image(thumbnail_path)
            pil_image.thumbnail((350, 350))
            # add thumbnail to widget
            pixmap = pil_to_pixmap(pil_image)
            label.setPixmap(pixmap)
        except Exception as e:
            label.setText(str(e))
        self.content.addWidget(label)

    def move_to_folder(self):
        # select folder
        folder = QFileDialog.getExistingDirectory(None, "Select Directory")
        if folder:
            file: FileRecord = self.data_context
            file_parent_folder_full_path = file.full_path
            query = get_file_record_by_folder(file_parent_folder_full_path)
            file_list = FileRecord.find(query)
            for file in file_list:
                file.move_to_folder(folder)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click event."""
        if event.button() == Qt.MouseButton.LeftButton:
            print("Label was double-clicked!")
            # Add any custom behavior you need her
            application: PySide6GlueApp = Allocator.get_instance(PySide6GlueApp)
            image_viwer = QuickImageViewer()
            application.show_window_modal(image_viwer)
            image_viwer.set_image(self.data_context.full_path)

    def on_annotate_button_click(self):
        file: FileRecord = self.data_context

        anotation_widget = Allocator.get_instance(Half_auto_editor_widget)
        job: AnnotationJob = anotation_widget.prop.current_job
        label = anotation_widget.job_label_dropdown.currentText()
        bg_task = Annotate_task(file, job, label)
        BGWorker.instance().add_task(bg_task)
        #job.annotate_file(file, label, override_annotation=True)
        anotation_widget.list_view.data_list.remove(file)


if __name__ == "__main__":
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    config.documentConfig.path = r"D:\data\ImageDataManager"
    QtApp = Half_auto_annotator_app()
    QtApp.set_main_widget(Half_auto_editor_widget())

    QtApp.run()
