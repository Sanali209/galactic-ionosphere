import os

from SLM.appGlue.DAL.DAL import GlueDataConverter
from SLM.appGlue.core import Allocator, GlueApp
from SLM.files_db.annotation_tool.annotation_export import DataSetExporterImageMultiClass_dirs_cum, \
    DataSetExporterImageMultiClass_anomali, DataSetExporterImageMultiClass_dirs

from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.files_db.components.fs_tag import TagRecord  # Added import

import flet as ft
from tqdm import tqdm

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo
from SLM.files_db.annotation_tool.annotation import SLMAnnotationClient, AnnotationJob

from SLM.destr_worck.bg_worcker import BGTask, BGWorker
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.flet.bindings.binds import BaseIterableToStrConverter
from SLM.flet.flet_ext import Flet_view, FletGlueApp, flet_dialog_alert
from SLM.flet.listView2 import ftListWidget, ListViewItemWidget

from applications.collectionTools.views.half_auto_annotation.annotation_prediction import AnnotationPredictionManager
import subprocess


# todo visualise progress

# todo add settings for annotators
# todo use of path of annotation
# todo write processing params in progress bar
# todo add capability for edit jobs (add, remove, edit)
# todo: add image samplers for random get or secvential order
class AnnotationRecord_FileRecord(GlueDataConverter):
    def Convert(self, data):
        return data.file

    def ConvertBack(self, data):
        return None


class Show_category_anotated(BGTask):
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
        gr_editor_ins: HalfAutoAnnotationViewV2 = Allocator.get_instance(HalfAutoAnnotationViewV2)
        if self.list_clear:
            gr_editor_ins.list_view.data_list.clear()
            self.all_items_count = len(self.items)

        if self.label is None:
            self.label = gr_editor_ins.job_label_dropdown.value
            if self.label is None:
                return
        curent_items = self.items[0:5000]
        next_items = self.items[5000:]
        converter = AnnotationRecord_FileRecord()
        for item in tqdm(curent_items):
            self.current_item_count += 1
            self.state.progress = self.current_item_count / self.all_items_count
            if item.value is None:
                continue
            if item.value != self.label:
                continue
            f_item = converter.Convert(item)
            gr_editor_ins.list_view.data_list.append(f_item)
            yield None
        if len(next_items) > 0:
            n_task = Show_category_anotated()
            n_task.all_items_count = self.all_items_count
            n_task.current_item_count = self.current_item_count
            n_task.items = next_items
            n_task.list_clear = False
            n_task.label = self.label
            BGWorker.instance().add_task(n_task, ignore_excludes=True)
        yield "done"


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
        gr_editor_ins: HalfAutoAnnotationViewV2 = Allocator.get_instance(HalfAutoAnnotationViewV2)
        if self.list_clear:
            gr_editor_ins.list_view.data_list.clear()
            self.all_items_count = len(self.items)

        if self.label is None:
            self.label = gr_editor_ins.job_label_dropdown.value
            if self.label is None:
                return
        annotator = AnnotationPredictionManager.instance().get_annotator_by_name(
            gr_editor_ins.prediction_piplain_dropdown.value)
        curent_items = self.items[0:5000]
        next_items = self.items[5000:]
        converter = AnnotationRecord_FileRecord()
        for item in tqdm(curent_items):
            try:
                self.current_item_count += 1
                self.state.progress = self.current_item_count / self.all_items_count

                f_item = converter.Convert(item)
                if annotator.is_satisfied_by(self.label, f_item):
                    gr_editor_ins.list_view.data_list.append(f_item)
                yield None
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                continue
        if len(next_items) > 0:
            n_task = Show_category_task()
            n_task.all_items_count = self.all_items_count
            n_task.current_item_count = self.current_item_count
            n_task.items = next_items
            n_task.list_clear = False
            n_task.label = self.label
            BGWorker.instance().add_task(n_task, ignore_excludes=True)
        yield "done"


class AnnotateTask(BGTask):
    def __init__(self, *args):
        super().__init__(*args)
        # logical error no prevent second execution
        item = self.args[0]
        self.name = "AnnotateTask" + str(item._id)
        self.exclude_names = [self.name]
        self.cancel_token = True

    def task_function(self, *args, **kwargs):
        item: FileRecord = self.args[0]
        job: AnnotationJob = self.args[1]
        label = self.args[2]
        list_view = self.args[3]
        job.annotate_file(item, label, override_annotation=True)
        list_view.data_list.remove(item)
        yield "done"


class lwItemTemplate(ListViewItemWidget):

    def build_header(self):
        self.width = 410
        file_record: FileRecord = self.data_context
        try:
            if file_record.gr_start:
                self.main_container.color = '#ff0000'

        except:
            pass
        name_text = ft.Text(file_record.name)
        row = ft.Row(wrap=True, spacing=2)
        annotate_button = ft.ElevatedButton(text="annotate", on_click=self.annotate)

        pb = ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(text="locate on explorer", on_click=self.locate_on_explorer),
                ft.PopupMenuItem(text="add parent folder to view", on_click=self.add_parent_folder_to_view),
                ft.PopupMenuItem(text="set reit hi", on_click=self.set_reit_hi),
                ft.PopupMenuItem(text="set reit low", on_click=self.set_reit_low),
                ft.PopupMenuItem(text="del", on_click=self.delete_from_view)
            ]
        )
        row.controls.extend([annotate_button, pb])
        try:
            thumb = file_record.get_thumb("medium")
            self.image = ft.Image(src=thumb, width=400, height=400, fit=ft.ImageFit.SCALE_DOWN
                                  )
        except Exception as e:
            self.image = ft.Text("image error")

        self.main_container.content = ft.Column([name_text, row, self.image], expand_loose=True, spacing=2,
                                                alignment=ft.alignment.center,
                                                horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def on_clicks(self, count):
        if count == 2:
            dialog = flet_dialog_alert("image", "")
            image = ft.Image(src=self.data_context.full_path, fit=ft.ImageFit.SCALE_DOWN)

            def on_ok_click(*args, **kwargs):
                dialog.close_dlg()

            dialog.set_content(
                ft.Column([
                    image,
                    ft.ElevatedButton(text="Ok", on_click=on_ok_click)
                ], width=1000)
            )
            dialog.show()

    def delete_from_view(self, event):
        self.parent_list_view.data_list.remove(self.data_context)

    def annotate(self, event):
        group_editor_view_in: HalfAutoAnnotationViewV2 = Allocator.get_instance(HalfAutoAnnotationViewV2)
        item: FileRecord = self.data_context
        job = group_editor_view_in.prop.current_job
        label = group_editor_view_in.job_label_dropdown.value

        task = AnnotateTask(item, job, label, self.parent_list_view)
        BGWorker.instance().add_task(task)

    def locate_on_explorer(self, event):
        # show selected item in explorer
        subprocess.run(["explorer", "/select,", os.path.normpath(self.data_context.full_path)], shell=True)
        #os.system(f"explorer /select,{self.data_context.full_path}")

    def add_parent_folder_to_view(self, event):
        # remove curent item from view
        self.parent_list_view.data_list.remove(self.data_context)
        # show selected item in explorer
        parent_folder = os.path.dirname(self.data_context.full_path)
        files = get_file_record_by_folder(parent_folder)
        for file in files:
            self.parent_list_view.data_list.remove(file)
            self.parent_list_view.data_list.append(file)

    def set_reit_hi(self, event):
        job_reit = AnnotationJob.get_by_name("rating")
        job_reit.annotate_file(self.data_context, "high", override_annotation=True)

    def set_reit_low(self, event):
        job_reit = AnnotationJob.get_by_name("rating")
        job_reit.annotate_file(self.data_context, "low", override_annotation=True)


class Show_anomaly(BGTask):
    def __init__(self):
        super().__init__()
        self.name = "show_anomaly_task"
        self.exclude_names = [self.name]

    def task_function(self, *args, **kwargs):
        gr_editor_ins: HalfAutoAnnotationViewV2 = Allocator.get_instance(HalfAutoAnnotationViewV2)
        gr_editor_ins.list_view.data_list.clear()
        label = gr_editor_ins.job_label_dropdown.value
        if label is None:
            return
        job = gr_editor_ins.prop.current_job
        items = job.get_ann_records_by_label(label)
        annotator = AnnotationPredictionManager.instance().get_annotator_by_name(
            gr_editor_ins.prediction_piplain_dropdown.value)
        for item in tqdm(items):
            if not annotator.is_satisfied_by(label, item.file):
                gr_editor_ins.list_view.data_list.append(item.file)
            yield None
        yield "done"


class Show_related_to(BGTask):
    def __init__(self):
        super().__init__()
        self.name = "show_anomaly_task"
        self.exclude_names = [self.name]

    def task_function(self, *args, **kwargs):
        gr_editor_ins: HalfAutoAnnotationViewV2 = Allocator.get_instance(HalfAutoAnnotationViewV2)
        gr_editor_ins.list_view.data_list.clear()
        label = gr_editor_ins.job_label_dropdown.value
        if label is None:
            return
        job = gr_editor_ins.prop.current_job
        items = job.get_ann_records_by_label(label)
        for item in tqdm(items):

            # get related to this file
            rel_filter = {"type": "similar_search",
                          "from_id": item.file._id,
                          }
            rel_records = RelationRecord.find(rel_filter)

            add_items = []
            for rel in rel_records:
                file = FileRecord(rel.to_id)
                annotation = job.get_annotation_record(file)
                if annotation is None:
                    continue
                if annotation.value is None:
                    if not gr_editor_ins.list_view.data_list.exist(file):
                        add_items.append(file)
            if len(add_items) > 0:

                gr_editor_ins.list_view.data_list.append(item.file)
                for sitem in add_items:
                    sitem.gr_start = True
                    gr_editor_ins.list_view.data_list.append(sitem)
            yield None
        yield "done"


class HalfAutoAnnotationViewV2(Flet_view):  # Renamed class
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

    progress_row: ft.Column
    center_column: ft.Column
    current_job_state_text: ft.Text
    active_job_dropdown: ft.Dropdown
    job_label_dropdown: ft.Dropdown
    source_path_text_field: ft.TextField
    prediction_piplain_dropdown: ft.Dropdown
    show_category_button: ft.ElevatedButton
    all_items_count_text: ft.Text
    current_page_text: ft.Text
    all_pages_count_text: ft.Text
    page_next_button: ft.ElevatedButton
    page_prev_button: ft.ElevatedButton

    def __init__(self):
        super().__init__()
        self.app: FletGlueApp = GlueApp.current_app
        self.prop = HalfAutoAnnotationViewV2.group_editor_bindings()  # Adjusted class name
        self.annotation_db = SLMAnnotationClient()
        Allocator.res.register(self)
        self.expand = True
        self.xml_source = """
        <root>
            <Row id="toolbar_row" >
                <MenuBar>
                    <SubmenuButton text="File">
                        <MenuItemButton text="Mark folder for annotation"  on_click="on_mark_folder_for_annotation_action"/>
                        <MenuItemButton text="clear for annotation" on_click="on_clear_for_annotation_action"/>
                        <MenuItemButton text="Backup Annotations" on_click="annotation_buckup"/>
                        <MenuItemButton text="Restore Annotations" on_click="annotation_restore"/>
                        <MenuItemButton text="Settings"  />
                        <SubmenuButton text="Export">
                            <MenuItemButton text="Export anomaly" on_click="export_anomaly"/>
                            <MenuItemButton text="Export Dataset (All Annotated)" on_click="on_export_dataset_all"/>
                        </SubmenuButton>
                    </SubmenuButton>
                    <SubmenuButton text="Edit">
                        <MenuItemButton text="Annotate page" on_click="page_annotate"  />
                        <MenuItemButton text="Show anotated" on_click="show_annotated"  />   
                        <MenuItemButton text="Get anomaly" on_click ="get_anomaly" />
                        <MenuItemButton text="Get related to label" on_click="get_related_to_label"/>
                        <MenuItemButton text="Create rel imageSet" on_click="create_relation_image_set"/>
                        <MenuItemButton text="Clear view" on_click="clear_view"/>
                        <MenuItemButton text="Import folder to view" on_click="import_folder_to_view"/>
                    </SubmenuButton>
                    <SubmenuButton text="Job">
                        <MenuItemButton text="Info" on_click="on_show_info"/>
                        <MenuItemButton text="Export to tags" on_click="on_export_tag"/>
                    </SubmenuButton>
                </MenuBar>
                <Button text="x" on_click="action_view_close"/>
            </Row>
            <Row expand="1">
               
                <Column id="center_column" expand="1">
                    <Row>
                        <Button id="page_next_button" text="n" on_click="on_page_next_button_click"/>
                        <Button id="page_prev_button" text="p" on_click="on_page_prev_button_click"/>
                        
                        
                        <Text id="all_pages_count_text" value="0"/>
                        <Text id="current_page_text" value="0"/>
                        <Text id="all_items_count_text" value="0"/>
                        
                    </Row>
                    <Row>
                    <Text id="current_job_state_text" value=""/>
                    <TextField id="source_path_text_field" label="source path"/>
                    <DropDown label="active job" id="active_job_dropdown"/>
                    <DropDown id="job_label_dropdown" label="job category"/>
                    <DropDown id="prediction_piplain_dropdown" label="prediction by"/>
                    <Button id="show_category_button" text="show" on_click="ShowCategory"/>
                    <Button id="stop_show_button" text="stop show" on_click="stop_show"/>
                    </Row>
                        
                </Column>  
            </Row>
            <Row id="footer_row" wrap="True">
                <ProgressBar id="gen_view_progress_bar" value="0.0" width="300"/>
                <Text id="gen_progress_message" value="message"/>
            </Row>
        </root>
        """
        self.parse_xml()

        # active job dropdown binding
        self.active_job_dropdown.options = [ft.dropdown.Option(text=str(x)) for x in self.prop.all_jobs]
        conv = BaseIterableToStrConverter(self.prop.all_jobs)
        self.prop.dispatcher.current_job.bind(bind_target=self.active_job_dropdown, converter=conv,
                                              field="value", callback=self.on_job_selected)
        # work path binding
        self.prop.dispatcher.current_work_path.bind(bind_target=self.source_path_text_field)

        # grid view
        self.list_view = ftListWidget()
        self.list_view.data_list_cursor.items_per_page = 100
        self.list_view.template.itemTemplateSelector.add_template(FileRecord, lwItemTemplate)
        self.center_column.controls.append(self.list_view)
        self.list_view.data_list_cursor.attach(self)
        self.stop_pending = False

        self.list_view.selection_mode = "multi"

    def show_annotated(self, event):
        task = Show_category_anotated()
        task.items = self.prop.current_job.get_all_annotated()
        BGWorker.instance().add_task(task)

    def create_relation_image_set(self, event):
        # create relation for selected items
        selection = self.list_view.get_selected()
        if len(selection) > 1:
            first = selection[0]
            for item in selection[1:]:
                rel = RelationRecord.set_relation(first, item, "similar_search")
                rel.set_field_val("sub_type", "similar")
                rel.set_field_val("distance", 0.99)
                rel.set_field_val("emb_type", "manual")
        self.list_view.select_clear()

    def on_mark_folder_for_annotation_action(self, *args, **kwargs):
        dialog = flet_dialog_alert("enter annotation path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            if self.prop.current_job is None: return
            self.prop.current_job.mark_not_annotated_in_directory(text_inputh.value)
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def on_clear_for_annotation_action(self, event):
        if self.prop.current_job is None: return
        self.prop.current_job.clear_not_annotated_list()

    def import_folder_to_view(self, event):
        dialog = flet_dialog_alert("enter folder path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            folder = text_inputh.value
            files = get_file_record_by_folder(folder, recurse=True)
            for file in files:
                self.list_view.data_list.append(file)
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def clear_view(self, event):
        self.list_view.data_list.clear()

    def export_anomaly(self, event):
        dialog = flet_dialog_alert("enter export path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            annotator = AnnotationPredictionManager.instance().get_annotator_by_name(
                self.prediction_piplain_dropdown.value)
            DataSetExporterImageMultiClass_anomali().ExportToDataset(text_inputh.value, self.prop.current_job,
                                                                     annotator)
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def get_anomaly(self, event):
        BGWorker.instance().add_task(Show_anomaly())

    def get_related_to_label(self, event):
        BGWorker.instance().add_task(Show_related_to())

    def page_annotate(self, event):
        current_page_items = self.list_view.get_current_page_data_items()
        for item in tqdm(current_page_items):
            item: FileRecord
            job = self.prop.current_job
            label = self.job_label_dropdown.value
            # do this in one task and optimize for speed
            task = AnnotateTask(item, job, label, self.list_view)
            BGWorker.instance().add_task(task)

    def action_view_close(self, event):
        self.close_view()

    def on_job_selected(self, *args, **kwargs):
        if self.prop.current_job is not None:
            # initialize job label dropdown
            labels = self.prop.current_job.choices
            if isinstance(labels, list) and len(labels) > 0:
                self.job_label_dropdown.options = [ft.dropdown.Option(text=x) for x in labels]
                self.job_label_dropdown.value = labels[0]

            # initialize prediction pipline dropdown
            an_pred_manager = AnnotationPredictionManager.instance()
            annotators = an_pred_manager.get_compatible_annotators(self.prop.current_job.name)
            if len(annotators) > 0:
                self.prediction_piplain_dropdown.options = [ft.dropdown.Option(text=x) for x in annotators]
                self.prediction_piplain_dropdown.value = annotators[0]

        self.update()

    def ShowCategory(self, event):
        task = Show_category_task()
        task.items = self.prop.current_job.get_all_not_annotated()
        BGWorker.instance().add_task(task)

    def stop_show(self, event):
        BGWorker.instance().cancel_task_by_names(["show_category_task", "show_anomaly_task"])

    def list_update(self, data_model, change_type, item=None):
        items_all_count = self.list_view.data_list_cursor.all_items_count()
        self.all_items_count_text.value = f"{items_all_count}"
        self.current_page_text.value = f"{self.list_view.data_list_cursor.current_page}"
        self.all_pages_count_text.value = f"{self.list_view.data_list_cursor.max_page}"
        if change_type == "refresh":

            pass
        elif change_type == "remove":
            # todo: realisation
            pass
        elif change_type == "clear":
            pass
        # !multi update?
        self.update()

    def on_page_next_button_click(self, event):
        self.list_view.data_list_cursor.page_next()
        self.update()

    def on_page_prev_button_click(self, event):
        self.list_view.data_list_cursor.page_previous()
        self.update()

    # Methods from single_anotation_flet.py
    def on_show_info(self, event):
        if self.prop.current_job is None:
            flet_dialog_alert("Error", "No job selected.").show()
            return
        dialog = flet_dialog_alert(f"Job Info: {self.prop.current_job.name}", "")
        list_str: list = self.prop.current_job.choices
        text_content = ""
        if not list_str:
            text_content = "No choices defined for this job."
        else:
            for label in list_str:
                count = self.prop.current_job.count_annotated_items(label)
                text_content += f"{label} : {count}\n"

        dialog.set_content(ft.Text(text_content))
        dialog.show()

    def on_export_tag(self, event):
        if self.prop.current_job is None:
            flet_dialog_alert("Error", "No job selected.").show()
            return

        item_list = self.prop.current_job.get_all_annotated()
        if not item_list:
            flet_dialog_alert("Info", "No items annotated in this job to export to tags.").show()
            return

        for item in tqdm(item_list, desc=f"Exporting tags for job '{self.prop.current_job.name}'"):
            file_record = item.file
            tag = TagRecord.get_or_create(f"annotation/{self.prop.current_job.name}/{item.value}")
            tag.add_to_file_rec(file_record)
        flet_dialog_alert("Success", f"Tags exported for job '{self.prop.current_job.name}'.").show()

    def annotation_buckup(self, *args, **kwargs):
        dialog = flet_dialog_alert("Enter backup path for annotations", "")
        text_inputh = ft.TextField(label="Path to JSON file")

        def on_ok_click(*args, **kwargs):
            path = text_inputh.value
            if not path:
                flet_dialog_alert("Error", "Path cannot be empty.").show()
                return

            client = SLMAnnotationClient()
            try:
                client.save_to_json(path)
                flet_dialog_alert("Success", f"Annotations backed up to {path}").show()
            except Exception as e:
                flet_dialog_alert("Error", f"Failed to backup annotations: {e}").show()
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path to save the annotation backup JSON file:"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def annotation_restore(self, *args, **kwargs):
        dialog = flet_dialog_alert("Enter path to restore annotations from", "")
        text_inputh = ft.TextField(label="Path to JSON file")

        def on_ok_click(*args, **kwargs):
            path = text_inputh.value
            if not path:
                flet_dialog_alert("Error", "Path cannot be empty.").show()
                return

            client = SLMAnnotationClient()
            try:
                client.restore_from_json(path)
                # Refresh job list and current job if necessary
                all_jobs = client.get_all_jobs({'type': "multiclass/image"})
                self.prop.all_jobs = all_jobs
                self.active_job_dropdown.options = [ft.dropdown.Option(text=str(x)) for x in self.prop.all_jobs]
                if self.prop.all_jobs:
                    self.prop.current_job = self.prop.all_jobs[0]
                    self.active_job_dropdown.value = str(self.prop.current_job)  # Trigger on_job_selected
                else:
                    self.prop.current_job = None
                    self.active_job_dropdown.value = None
                self.on_job_selected()  # Manually call to refresh UI elements dependent on job
                flet_dialog_alert("Success", f"Annotations restored from {path}").show()
            except Exception as e:
                flet_dialog_alert("Error", f"Failed to restore annotations: {e}").show()
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path of the annotation backup JSON file to restore:"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def on_export_dataset_all(self, event):
        if self.prop.current_job is None:
            flet_dialog_alert("Error", "No job selected.").show()
            return

        dialog = flet_dialog_alert("Enter export path for dataset", "")
        text_inputh = ft.TextField(label="Directory path for export")

        def on_ok_click(*args, **kwargs):
            export_path = text_inputh.value
            if not export_path:
                flet_dialog_alert("Error", "Export path cannot be empty.").show()
                return

            try:
                DataSetExporterImageMultiClass_dirs().ExportToDataset(export_path, self.prop.current_job)
                flet_dialog_alert("Success",
                                  f"Dataset exported to {export_path} for job '{self.prop.current_job.name}'.").show()
            except Exception as e:
                flet_dialog_alert("Error", f"Failed to export dataset: {e}").show()
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text(f"Enter directory path to export dataset for job '{self.prop.current_job.name}':"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()


if __name__ == "__main__":
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    app = FletGlueApp()
    view = HalfAutoAnnotationViewV2()  # Adjusted class name
    app.set_view(view)

    app.run()
