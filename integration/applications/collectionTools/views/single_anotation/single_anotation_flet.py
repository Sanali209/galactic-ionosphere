import flet as ft
from tqdm import tqdm

from SLM.appGlue.DAL.binding.bind import PropInfo, PropUser
from SLM.appGlue.core import Allocator

from SLM.files_db.annotation_tool.annotation import SLMAnnotationClient, AnnotationJob
from SLM.files_db.annotation_tool.annotation_export import DataSetExporterImageMultiClass_dirs
from SLM.files_db.components.File_record_wraper import FileRecord

from SLM.files_db.components.fs_tag import TagRecord
from SLM.flet.bindings.binds import BaseIterableToStrConverter
from SLM.flet.flet_ext import Flet_view, FletGlueApp, flet_dialog_alert
from SLM.flet.object_editor import FtObjectEditor, ObjectEditorView

from SLM.groupcontext import group



# todo add job editor
# add options for re annotate

class group_editor_bindings(PropUser):
    all_jobs: list[AnnotationJob] = PropInfo()
    current_job: AnnotationJob = PropInfo()
    currentImageRecord: FileRecord = PropInfo()

    def __init__(self):
        super().__init__()
        annotation_db = SLMAnnotationClient()
        all_jobs = annotation_db.get_all_jobs({'type': "multiclass/image"})
        self.all_jobs = all_jobs
        valg = self.all_jobs
        if len(valg) > 0:
            self.current_job = self.all_jobs[0]


class SingleAnnotationView(Flet_view):
    left_column: ft.Column
    center_column: ft.Column
    chooses_column: ft.Column
    active_job_dropdown: ft.Dropdown
    cur_state_progress: ft.Text
    add_folder_button: ft.ElevatedButton
    next_button: ft.ElevatedButton
    prev_button: ft.ElevatedButton
    annotate_button: ft.ElevatedButton

    def __init__(self):
        super().__init__()
        self.prop = group_editor_bindings()
        self.annotation_db = SLMAnnotationClient()
        self.expand = True
        self.xml_source = """
        <root>
            <Row id="toolbar_row">
            <MenuBar>
                    <SubmenuButton text="File">
                        <MenuItemButton text="Mark folder for annotation"  on_click="on_mark_folder_for_annotation_action"/>
                        <MenuItemButton text="clear for annotation" on_click="on_clear_for_annotation_action"/>
                        <MenuItemButton text="Backup" on_click="annotation_buckup"/>
                        <MenuItemButton text="Restore" on_click="annotation_restore"/>
                        <SubmenuButton text="Export">
                            <MenuItemButton text="Restore" on_click="on_export"/>
                        </SubmenuButton>
                    </SubmenuButton>
                    <SubmenuButton text="Job">
                        <MenuItemButton text="Info" on_click="on_show_info"/>
                        <MenuItemButton text="Export to tags" on_click="on_export_tag"/>
                    </SubmenuButton>
            </MenuBar>
                <Button text="x" on_click="action_view_close"/>
            </Row>
            <Row expand="1">
                <Column id="left_column" expand="1">
                    <DropDown label="active job" id="active_job_dropdown"/>
                        <Text id="cur_state_progress" value = "Current job state"/>
                        <Row>
                        <Button text="next" id="next_button" on_click="on_next_button_click"/>
                        <Button text="prev" id="prev_button" on_click="on_prev_button_click"/>
                        <Button text="annotate" id="annotate_button" on_click="on_annotate_button_click"/>
                        
                        </Row>
                    <Text value = "Chooses"/>
                        <Column id="chooses_column">
                        
                        </Column>
                </Column>
                
                
                <Column id="center_column" expand="3">
                </Column>
            </Row>
        </root>
        """
        self.cached_list = []
        self.cached_list_count = 0
        self.cached_list_cur = 0

        self.parse_xml()

        with group():  # left column
            # todo: adjast column placing
            self.chooses_radio = ft.RadioGroup()
            self.chooses_column.controls.append(self.chooses_radio)
            self.chooses_column.height = 500
            self.chooses_column.scroll = ft.ScrollMode.ADAPTIVE

        with group():
            #self.center_column.scroll = ft.ScrollMode.ALWAYS
            self.prop_editor = FtObjectEditor()
            self.center_column.controls.append(self.prop_editor)
            self.prop_editor.add_view_template(
                lambda obj: fsImageRecordPropView if isinstance(obj, FileRecord) else None)
        # initialize controls
        self.active_job_dropdown.options = [ft.dropdown.Option(text=str(x)) for x in self.prop.all_jobs]
        conv = BaseIterableToStrConverter(self.prop.all_jobs)
        self.prop.dispatcher.current_job.bind(bind_target=self.active_job_dropdown, converter=conv,
                                              callback=self.on_job_selected, field="value")

    def action_view_close(self, event):
        self.close_view()

    def on_export_tag(self, event):
        item_list = self.prop.current_job.get_all_annotated()
        for item in tqdm(item_list):
            file_record = item.file
            tag = TagRecord.get_or_create(f"annotation/{self.prop.current_job.name}/{item.value}")
            tag.add_to_file_rec(file_record)

    def on_show_info(self, event):
        dialog = flet_dialog_alert("job info", "")
        list_str: list = self.prop.current_job.choices
        text = ""
        for label in list_str:
            count = self.prop.current_job.count_annotated_items(label)
            text += f"{label} : {count}\n"
        dialog.set_content(ft.Text(text))
        dialog.show()

    def on_export(self, event):
        dialog = flet_dialog_alert("enter export path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            DataSetExporterImageMultiClass_dirs().ExportToDataset(text_inputh.value, self.prop.current_job, )
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def on_keyboard_event(self, event):
        # todo implement key map and key map editor
        if event.key == " ":
            # not proper way to handle event
            #self.on_annotate_button_click()
            pass

    def on_job_selected(self, *args, **kwargs):
        if self.prop.current_job is None:
            return
        choices = self.prop.current_job.choices
        self.chooses_radio.content = ft.Column([
            ft.Radio(value=choice, label=choice) for choice in choices
        ])
        self.cached_list = self.prop.current_job.get_all_not_annotated()
        self.cached_list_count = len(self.cached_list)
        self.cached_list_cur = 0
        self.set_current_image()
        self.update()

    def on_next_button_click(self, *args, **kwargs):
        if self.prop.current_job is not None:
            self.cached_list_cur += 1
            self.set_current_image()
            self.update()

    def on_prev_button_click(self, *args, **kwargs):
        if self.prop.current_job is not None:
            self.cached_list_cur -= 1
            self.set_current_image()
            self.update()

    def on_annotate_button_click(self, *args, **kwargs):
        # treed unsafe
        if self.prop.current_job is not None:
            if self.chooses_radio.value is not None:
                curent_annotation = self.cached_list[self.cached_list_cur]
                curent_annotation.value = self.chooses_radio.value
                self.cached_list_cur += 1
                self.set_current_image()
                self.update()

    def set_current_image(self):
        if self.prop.current_job is not None:

            try:
                current_item = self.cached_list[self.cached_list_cur]
            except IndexError:
                current_item = None

            if current_item is not None:

                self.cur_state_progress.spans = [
                    ft.TextSpan(f"current item: {self.cached_list_cur}/"
                                f"{self.cached_list_count}")]
                fileRecord = current_item.file
                self.prop_editor.set_object(fileRecord)
                #self.current_annotation_image.src = os.path.join(current_item.full_path)
                anotat_record = self.prop.current_job.get_annotation_record(current_item)
                if anotat_record is not None:
                    item_annotation_value = anotat_record.value
                    self.chooses_radio.value = item_annotation_value
            else:
                self.cur_state_progress.spans = [
                    ft.TextSpan(f"current item: 0/"
                                f"0")]
                self.prop_editor.clear()
        self.update()

    def on_mark_folder_for_annotation_action(self, *args, **kwargs):
        dialog = flet_dialog_alert("enter annotation path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            if self.prop.current_job is None: return
            self.prop.current_job.mark_not_annotated_in_directory(text_inputh.value)
            self.set_current_image()
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

    def annotation_restore(self, *args, **kwargs):
        dialog = flet_dialog_alert("enter annotation path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            if self.prop.current_job is None: return
            path = text_inputh.value
            client = SLMAnnotationClient()
            # todo process entered path for more easy use(remove " and other)
            client.restore_from_json(path)
            self.set_current_image()
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()

    def annotation_buckup(self, *args, **kwargs):
        dialog = flet_dialog_alert("enter annotation path", "")
        text_inputh = ft.TextField()

        def on_ok_click(*args, **kwargs):
            if self.prop.current_job is None: return
            path = text_inputh.value
            client = SLMAnnotationClient()
            client.save_to_json(path)
            self.set_current_image()
            dialog.close_dlg()

        dialog.set_content(
            ft.Column([
                ft.Text("Enter path"),
                text_inputh,
                ft.ElevatedButton(text="Ok", on_click=on_ok_click)
            ], width=500)
        )
        dialog.show()


class fsImageRecordViewProps(PropUser):
    pass


class fsImageRecordPropView(ObjectEditorView):
    image_record_view: ft.Column
    annotate_image: ft.Image

    def __init__(self, sel_object: object = None):
        super().__init__(sel_object)

        self.prop = fsImageRecordViewProps()
        self.object: FileRecord = sel_object
        self.xml_source = """
        <root>
            <Column id="image_record_view" >
                <Text value = "read:self.object.full_path" />
                <Image id="annotate_image" src = "read:self.object.full_path"/>
                
            </Column>
        </root>
        """

        self.parse_xml()

        self.image_record_view.expand = True
        self.image_record_view.scroll = ft.ScrollMode.AUTO


if __name__ == "__main__":
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    app = FletGlueApp()
    view = SingleAnnotationView()
    app.set_view(view)

    app.run()
