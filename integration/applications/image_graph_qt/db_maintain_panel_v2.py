import os
import re
from tqdm import tqdm

from PySide6.QtWidgets import (QVBoxLayout, QLineEdit, QTabWidget, QWidget,
                               QPushButton, QDialog, QLabel, QProgressBar,
                               QApplication, QFormLayout, QDialogButtonBox,
                               QComboBox, QDoubleSpinBox, QCheckBox, QGroupBox)
from loguru import logger

from SLM.appGlue.core import Allocator
from SLM.appGlue.progress_visualize import ProgressManager, ProgressVisualizer
from SLM.files_db.annotation_tool.annotation import AnnotationJob
from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.files_db.components.relations.dubsearch import create_graf_dubs, del_image_search_refs
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.files_functions.index_folder import index_folder_one_thread
from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget, PySide6GlueApp
from SLM.vision.imagetotensor.CNN_Encoding import ImageToCNNTensor

# ##################################################################
# Progress Dialog
# ##################################################################
class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing...")
        self.setModal(True)
        self.layout = QVBoxLayout(self)
        self.description_label = QLabel("Description: Initializing...")
        self.description_label.setWordWrap(True)
        self.layout.addWidget(self.description_label)
        self.message_label = QLabel("Status: Please wait...")
        self.message_label.setWordWrap(True)
        self.layout.addWidget(self.message_label)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)
        self.setMinimumWidth(800)
        self.setMinimumHeight(150)

    def update_dialog(self, description, message, current_value, max_value):
        if max_value > 10000 and current_value % 10 != 0:
            return
        self.description_label.setText(f"Task: {description}")
        self.message_label.setText(f"Status: {message}")
        self.progress_bar.setMaximum(max_value if max_value > 0 else 100)
        self.progress_bar.setValue(current_value)
        QApplication.processEvents()

    def closeEvent(self, event):
        event.accept()

# ##################################################################
# Settings Dialogs
# ##################################################################
class FindDuplicatesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find New Duplicates Settings")
        self.layout = QFormLayout(self)

        self.path_edit = QLineEdit()
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(0.4)
        self.pats_dubs_search_checkbox = QCheckBox()
        self.pats_dubs_search_checkbox.setChecked(True)
        self.related_search_checkbox = QCheckBox()

        self.layout.addRow("Folder Path:", self.path_edit)
        self.layout.addRow("Similarity Threshold:", self.threshold_spinbox)
        self.layout.addRow("Search within Path:", self.pats_dubs_search_checkbox)
        self.layout.addRow("Related Search:", self.related_search_checkbox)

        self.encoders_group = QGroupBox("Encoders")
        self.encoders_layout = QVBoxLayout()
        self.encoders_group.setLayout(self.encoders_layout)
        self.encoder_checkboxes = {}
        for encoder_name in ImageToCNNTensor.all_backends.keys():
            checkbox = QCheckBox(encoder_name)
            self.encoder_checkboxes[encoder_name] = checkbox
            self.encoders_layout.addWidget(checkbox)
        self.layout.addRow(self.encoders_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addRow(self.button_box)

    def get_settings(self):
        selected_encoders = [name for name, cb in self.encoder_checkboxes.items() if cb.isChecked()]
        return {
            "path": self.path_edit.text(),
            "threshold": self.threshold_spinbox.value(),
            "pats_dubs_search": self.pats_dubs_search_checkbox.isChecked(),
            "related_search": self.related_search_checkbox.isChecked(),
            "encoders": selected_encoders
        }

class BulkDeleteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk Delete Relations")
        self.layout = QFormLayout(self)
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(0.5)
        self.layout.addRow("Delete relations with distance >", self.threshold_spinbox)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addRow(self.button_box)

    def get_threshold(self):
        return self.threshold_spinbox.value()

class ChangeDriveLetterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Drive Letter / Base Path")
        self.layout = QFormLayout(self)
        self.from_path_edit = QLineEdit()
        self.to_path_edit = QLineEdit()
        self.layout.addRow("From Path:", self.from_path_edit)
        self.layout.addRow("To Path:", self.to_path_edit)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addRow(self.button_box)

    def get_paths(self):
        return self.from_path_edit.text(), self.to_path_edit.text()

# ##################################################################
# Main Application Widget
# ##################################################################
class DBMaintainView(PySide6GlueWidget, ProgressVisualizer):
    def __init__(self):
        Allocator.res.register(self)
        super().__init__()
        prog_man = ProgressManager.instance()
        prog_man.add_visualizer(self)
        self.progress_dialog = ProgressDialog(self)

    def define_gui(self):
        self.setWindowTitle("DB Maintenance Panel")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self._create_files_folders_tab()
        self._create_duplicates_relations_tab()
        self._create_annotations_tab()
        self._create_thumbnails_cache_tab()

    def update_progress(self):
        prog_man = ProgressManager.instance()
        self.progress_dialog.update_dialog(
            prog_man.description, prog_man.message, prog_man.progress, prog_man.max_progress
        )

    def run_task_with_progress(self, task_function, description, *args, **kwargs):
        prog_man = ProgressManager.instance()
        prog_man.reset()
        prog_man.set_description(description)
        self.progress_dialog.update_dialog(description, "Initializing...", 0, 100)
        self.progress_dialog.show()
        QApplication.processEvents()
        try:
            task_function(*args, **kwargs)
            prog_man.step("Completed.")
        except Exception as e:
            logger.error(f"Error during {description}: {e}")
            prog_man.step(f"Error: {e}")
        finally:
            QApplication.processEvents()
            self.progress_dialog.hide()

    # ------------------- TAB CREATION -------------------
    def _create_files_folders_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tab_widget.addTab(tab, "Files & Folders")

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Enter folder path...")
        layout.addWidget(self.folder_input)

        layout.addWidget(QPushButton("Add Folder to DB", clicked=self.add_folder_action))
        layout.addWidget(QPushButton("Index Folder", clicked=self.index_folder_action))
        layout.addWidget(QPushButton("Delete Non-Existent Files from DB", clicked=self.delete_non_existent_action))
        
        self.ext_filter_input = QLineEdit(".ini")
        layout.addWidget(self.ext_filter_input)
        layout.addWidget(QPushButton("Delete Files by Extension", clicked=self.delete_by_ext_action))
        
        layout.addWidget(QPushButton("Change Drive Letter", clicked=self.change_drive_letter_action))
        layout.addStretch()

    def _create_duplicates_relations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tab_widget.addTab(tab, "Duplicates & Relations")

        layout.addWidget(QPushButton("Find New Duplicates...", clicked=self.find_new_duplicates_action))
        layout.addWidget(QPushButton("Optimize Relations (Clean Broken)", clicked=self.optimize_relations_action))
        layout.addWidget(QPushButton("Bulk Delete Relations by Threshold", clicked=self.bulk_delete_relations_action))
        layout.addStretch()

    def _create_annotations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tab_widget.addTab(tab, "Annotations")
        
        self.ann_job_combo = QComboBox()
        self.load_annotation_jobs()
        layout.addWidget(self.ann_job_combo)

        layout.addWidget(QPushButton("Remove Duplicate Annotations", clicked=self.remove_duplicate_annotations_action))
        layout.addWidget(QPushButton("Remove Broken Annotations", clicked=self.remove_broken_annotations_action))
        layout.addWidget(QPushButton("Clear All Annotations from Job", clicked=self.clear_annotation_job_action))
        layout.addWidget(QPushButton("Rename Label in Job", clicked=self.rename_label_action))
        layout.addStretch()

    def _create_thumbnails_cache_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tab_widget.addTab(tab, "Thumbnails & Cache")

        layout.addWidget(QPushButton("Refresh All Thumbnails", clicked=self.refresh_all_thumbnails_action))
        layout.addWidget(QPushButton("Clear All Face Detections", clicked=self.clear_all_face_detections_action))
        layout.addStretch()

    # ------------------- ACTIONS & LOGIC -------------------
    # Files & Folders Tab
    def add_folder_action(self):
        path = self.folder_input.text()
        if path:
            self.run_task_with_progress(FileRecord.add_file_records_from_folder, "Adding Folder", path)

    def index_folder_action(self):
        path = self.folder_input.text()
        if path:
            self.run_task_with_progress(index_folder_one_thread, "Indexing Folder", path)

    def delete_non_existent_action(self):
        path = self.folder_input.text()
        if path:
            self.run_task_with_progress(self._delete_non_existent_logic, "Deleting Non-Existent", path)

    def _delete_non_existent_logic(self, folder_path):
        prog_man = ProgressManager.instance()
        files = list(get_file_record_by_folder(folder_path, recurse=True))
        prog_man.max_progress = len(files) if files else 1
        for i, file_obj in enumerate(files):
            prog_man.step(f"Checking {file_obj.full_path or 'Unknown path'} ({i + 1}/{prog_man.max_progress})")
            if file_obj.full_path is None or not os.path.exists(file_obj.full_path):
                logger.warning(f"File not found, deleting record: {file_obj.full_path}")
                file_obj.delete()

    def delete_by_ext_action(self):
        ext = self.ext_filter_input.text()
        if ext:
            self.run_task_with_progress(self._delete_by_ext_logic, f"Deleting by Extension '{ext}'", ext)

    def _delete_by_ext_logic(self, ext):
        prog_man = ProgressManager.instance()
        query = {'name': {"$regex": '.*' + re.escape(ext) + '$'}}
        records_list = list(FileRecord.find(query))
        prog_man.max_progress = len(records_list) if records_list else 1
        for i, record in enumerate(records_list):
            prog_man.step(f"Deleting {record.name} ({i + 1}/{prog_man.max_progress})")
            record.delete_rec()

    def change_drive_letter_action(self):
        dialog = ChangeDriveLetterDialog(self)
        if dialog.exec():
            from_path, to_path = dialog.get_paths()
            if from_path and to_path:
                self.run_task_with_progress(self._change_drive_letter_logic, "Changing Drive Letters", from_path, to_path)

    def _change_drive_letter_logic(self, from_path, to_path):
        prog_man = ProgressManager.instance()
        files = list(FileRecord.find({'local_path': {"$regex": f"^{re.escape(from_path)}"}}))
        prog_man.max_progress = len(files)
        for i, file in enumerate(files):
            new_path = file.local_path.replace(from_path, to_path, 1)
            prog_man.step(f"Updating {file.name} to {new_path} ({i+1}/{len(files)})")
            file.local_path = new_path
            file.save()

    # Duplicates & Relations Tab
    def find_new_duplicates_action(self):
        dialog = FindDuplicatesDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            if settings["path"] and settings["encoders"]:
                self.run_task_with_progress(self._find_new_duplicates_logic, "Finding New Duplicates", settings)

    def _find_new_duplicates_logic(self, settings):
        prog_man = ProgressManager.instance()
        path = settings["path"]
        encoders = settings["encoders"]
        prog_man.max_progress = len(encoders)
        for i, encoder in enumerate(encoders):
            prog_man.step(f"Running encoder {encoder} ({i+1}/{len(encoders)})")
            create_graf_dubs(
                [path],
                related=[],
                distance=settings["threshold"],
                encoder=encoder,
                pats_dubs_search=settings["pats_dubs_search"],
                related_search=settings["related_search"]
            )

    def optimize_relations_action(self):
        self.run_task_with_progress(self._optimize_relations_logic, "Optimizing Relations")

    def _optimize_relations_logic(self):
        prog_man = ProgressManager.instance()
        relations = list(RelationRecord.find({'type': "similar_search"}))
        prog_man.max_progress = len(relations)
        for i, relation in enumerate(relations):
            prog_man.step(f"Checking relation {i+1}/{len(relations)}")
            from_rec = FileRecord.get_by_id(relation.from_id)
            to_rec = FileRecord.get_by_id(relation.to_id)
            if from_rec is None or to_rec is None or not from_rec.exists() or not to_rec.exists():
                relation.delete_rec()
                continue
            if from_rec.full_path == to_rec.full_path:
                relation.delete_rec()

    def bulk_delete_relations_action(self):
        dialog = BulkDeleteDialog(self)
        if dialog.exec():
            threshold = dialog.get_threshold()
            self.run_task_with_progress(del_image_search_refs, "Bulk Deleting Relations", threshold)

    # Annotations Tab
    def load_annotation_jobs(self):
        self.ann_job_combo.clear()
        jobs = AnnotationJob.find({})
        for job in jobs:
            self.ann_job_combo.addItem(job.name, job)

    def get_selected_job(self):
        return self.ann_job_combo.currentData()

    def remove_duplicate_annotations_action(self):
        job = self.get_selected_job()
        if job:
            self.run_task_with_progress(job.remove_annotation_dublicates2, f"Removing duplicates from {job.name}")

    def remove_broken_annotations_action(self):
        job = self.get_selected_job()
        if job:
            self.run_task_with_progress(job.remove_broken_annotations, f"Removing broken from {job.name}")

    def clear_annotation_job_action(self):
        from PySide6.QtWidgets import QMessageBox
        job = self.get_selected_job()
        if job:
            reply = QMessageBox.question(self, 'Confirm Deletion',
                                         f"This will delete ALL annotations from the job '{job.name}'. This cannot be undone. Are you sure?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.run_task_with_progress(job.clear_job, f"Clearing job {job.name}")

    def rename_label_action(self):
        logger.info("Rename Label action triggered.")

    # Thumbnails & Cache Tab
    def refresh_all_thumbnails_action(self):
        self.run_task_with_progress(self._refresh_all_thumbnails_logic, "Refreshing All Thumbnails")

    def _refresh_all_thumbnails_logic(self):
        prog_man = ProgressManager.instance()
        all_files = list(FileRecord.find({}))
        prog_man.max_progress = len(all_files)
        for i, file_rec in enumerate(all_files):
            prog_man.step(f"Refreshing {file_rec.name} ({i+1}/{len(all_files)})")
            file_rec.refresh_thumb()

    def clear_all_face_detections_action(self):
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     "This will delete ALL face detection data and face-based relations. This cannot be undone. Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.run_task_with_progress(self._clear_all_face_detections_logic, "Clearing All Face Detections")

    def _clear_all_face_detections_logic(self):
        from SLM.files_db.object_recognition.object_recognition import Detection
        prog_man = ProgressManager.instance()

        # Delete all detection objects
        detections = list(Detection.find({}))
        prog_man.max_progress = len(detections)
        for i, det in enumerate(detections):
            prog_man.step(f"Deleting detection object {i+1}/{len(detections)}")
            det.delete_rec()

        # Delete all face search relations
        relations = list(RelationRecord.find({'type': "similar_face_search"}))
        prog_man.max_progress += len(relations)
        for i, rel in enumerate(relations):
            prog_man.step(f"Deleting face relation {i+1}/{len(relations)}")
            rel.delete_rec()

        # Remove indexed_by flag from all files
        files = list(FileRecord.find({"indexed_by": "face_detection"}))
        prog_man.max_progress += len(files)
        for i, file in enumerate(files):
            prog_man.step(f"Updating file record {i+1}/{len(files)}")
            file.list_remove("indexed_by", "face_detection")


# ##################################################################
# Main execution
# ##################################################################
if __name__ == '__main__':
    class DBMaintainApp(PySide6GlueApp):
        pass

    config = Allocator.config
    Allocator.disable_module("VisionModule")
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"


    QtApp = DBMaintainApp()
    main_view = DBMaintainView()
    QtApp.set_main_widget(main_view)
    QtApp.run()
