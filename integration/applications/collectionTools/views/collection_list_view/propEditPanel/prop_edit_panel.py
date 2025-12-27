from tqdm import tqdm

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo
from SLM.appGlue.DesignPaterns.SingletonAtr import singleton
from SLM.files_db.annotation_tool.annotation import SLMAnnotationClient, AnnotationJob
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.object_recognition.object_recognition import Detection

from SLM.files_db.components.fs_tag import TagRecord
from SLM.flet.object_editor import ObjectEditorView, FtObjectEditor, FtObjectEditorTemplate
import flet as ft

from applications.collectionTools.views.collection_list_view.selectionManager.selection_manager import SelectionManager
from applications.collectionTools.views.collection_list_view.view_controller import ViewController


@singleton
class prop_editor_model:
    def __init__(self):
        self.prop_editor = FtObjectEditor()
        self.mode_select_control = ft.Dropdown(label="mode", options=[ft.dropdown.Option("properties"),
                                                                      ft.dropdown.Option("annotation")],
                                               value="properties", on_change=self.on_mode_change)
        self.properties_mode_template = FtObjectEditorTemplate()
        self.properties_mode_template.item_view_template.add_template_selector(
            lambda obj: fsFileRecordPropEditorView if obj is not None and len(obj) > 0
                                                      and isinstance(obj[0], FileRecord) else None
        )
        self.properties_mode_template.item_view_template.add_template_selector(
            lambda obj: DetectionPropEditView if obj is not None and len(obj) > 0
                                                 and isinstance(obj[0], Detection) else None
        )

        self.prop_editor.switch_editor_template(self.properties_mode_template)

        self.annotation_mode_template = FtObjectEditorTemplate()
        self.annotation_mode_template.item_view_template.add_template_selector(
            lambda obj: fsFileRecordAnnotateEditorView if obj is not None and len(obj) > 0 and isinstance(obj[0],
                                                                                                          FileRecord) else None
        )
        SelectionManager().register_on_selection_changed(self.on_item_selected)

    def on_item_selected(self, selection,*args, **kwargs):
        prop_editor_model().prop_editor.set_object(selection)

    def on_mode_change(self, event):
        str_value = event.data
        if str_value == "properties":
            self.prop_editor.switch_editor_template(self.properties_mode_template)
        elif str_value == "annotation":
            self.prop_editor.switch_editor_template(self.annotation_mode_template)


class fsFileRecordPropEditorView(ObjectEditorView):
    class fsFileRecordViewProps(PropUser):
        title_prop: str = PropInfo()
        description_prop: str = PropInfo()

        def __init__(self, file_record: FileRecord):
            super().__init__()
            self.title_prop = file_record.title
            self.description_prop = file_record.description
            self.dispatcher.description_prop.add_listener(
                lambda value: setattr(file_record,"description",value))

    tags_text: ft.Text

    def __init__(self, sel_object: object = None):
        super().__init__(sel_object)
        self.selection_list: list = sel_object
        self.file_record: FileRecord = self.selection_list[0]
        self.prop = fsFileRecordPropEditorView.fsFileRecordViewProps(self.file_record)
        """
        
        <Text value = "read:self.object.rating" />
        
        <Text value = "read:self.object.notes" />
        """
        self.xml_source = """
        <root>
                <Text value="full path"/>
                <Text value = "read:self.file_record.full_path" />
                <TextField  value = "bind:title_prop" label="title"/>
                <TextField value="bind:description_prop" label="description" multiline="True"/>
                <Button text="get ai descr" on_click="on_get_ai_des"/>
                <Text value="Tags:"/>
                <Text id="tags_text" />
        </root>
        """
        self.parse_xml()
        tags_recs = TagRecord.get_tags_of_file(self.file_record)
        self.tags_text.value = ", ".join([tag.name for tag in tags_recs])
        self.prop.dispatcher.title_prop.add_listener(self.on_title_prop_change)

    def on_get_ai_des(self, event):
        val = self.file_record.get_ai_expertise("image-text",
                                                "text_salesforce_blip_image_base")
        self.prop.description_prop = val['data']

    def on_title_prop_change(self, value):
        # todo write value to all selection
        self.file_record.title = value


# for annotating files
class fsFileRecordAnnotateEditorView(ObjectEditorView):
    class fsFileRecordViewProps(PropUser):
        def __init__(self, file_record: FileRecord):
            super().__init__()

    def __init__(self, edit_object: object = None):
        super().__init__(edit_object)
        self.selection_list: list = edit_object
        self.file_record: FileRecord = self.selection_list[0]
        self.prop = self.__class__.fsFileRecordViewProps(self.file_record)
        self.xml_source = """
        <root>
                <Text value="Full path:"/>
                <Text value = "read:self.file_record.full_path" />
        </root>
        """
        self.parse_xml()

        annotation_db = SLMAnnotationClient()
        all_jobs = annotation_db.get_all_jobs({'type': "multiclass/image"})
        for job in all_jobs:
            self.controls.append(ft.Text(job.name))
            dr_down = ft.Dropdown(label=job.name, on_change=self.on_annotation_change)
            self.controls.append(dr_down)
            dr_down.job = job
            dr_down.options = [ft.dropdown.Option(val) for val in list(job.choices)]
            an_record = job.get_annotation_record(self.file_record)
            if an_record is not None:
                dr_down.value = an_record.value

    def on_annotation_change(self, event):
        # todo implement bach annotation as on json import
        # block interface on time of annotating
        control = event.control
        job: AnnotationJob = control.job
        label = event.data
        for coll_record in tqdm(self.selection_list):
            job.annotate_file(coll_record, label, True)


class DetectionPropEditView(ObjectEditorView):
    class Props(PropUser):
        def __init__(self, edit_object):
            super().__init__()

    def __init__(self, edit_object: object = None):
        super().__init__(edit_object)
        self.selection_list: list = edit_object
        self.detection_record: Detection = self.selection_list[0]
        self.prop = self.__class__.Props(edit_object)
        self.xml_source = """
        <root>
            <Text value = "read:self.detection_record._id"/>
            <Text value = "read:self.detection_record.object_class" />
            <TextField label="name" value=""/>
            <Button text="delete" on_click="on_delete"/>
        </root>
        """
        self.parse_xml()

    def on_annotation_change(self, event):
        # todo implement bach annotation as on json import
        # block interface on time of annotating
        control = event.control
        job: AnnotationJob = control.job
        label = event.data
        for coll_record in tqdm(self.selection_list):
            job.annotate_file(coll_record, label, True)
        pass

    def on_delete(self, event):
        ViewController.delete_Detection(self.detection_record)


class RecognizedObjectPropEditView(ObjectEditorView):
    class Props(PropUser):
        recognized_name: str = PropInfo()

        def __init__(self, edit_object):
            super().__init__()
            self.recognized_name = edit_object.obj_class

    def __init__(self, edit_object: object = None):
        super().__init__(edit_object)
        self.selection_list: list = edit_object
        self.detection_record: Detection = self.selection_list[0]
        self.prop = self.__class__.Props(self.file_record)
        self.xml_source = """
        <root>
            <Text value="read:detection_record.obj_class"/>
            <TextField label="name" value="bind:recognized_name"/>
            <Button text="rename" on_click="self.on_recognition_rename"/>
        </root>
        """
        self.parse_xml()

    def on_annotation_change(self, event):
        # todo implement bach annotation as on json import
        # block interface on time of annotating
        control = event.control
        job: AnnotationJob = control.job
        label = event.data
        for coll_record in tqdm(self.selection_list):
            job.annotate_file(coll_record, label, True)
