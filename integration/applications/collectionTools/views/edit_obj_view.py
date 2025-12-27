import flet as ft

from SLM.appGlue.DAL.binding.bind import PropUser, PropInfo

from SLM.appGlue.progress_visualize import ProgressManager
from SLM.files_db.components.File_record_wraper import FileRecord

from SLM.flet.flet_ext import Flet_view, FletGlueApp
from SLM.flet.object_editor import FtObjectEditor, ObjectEditorView
from SLM.vision.imagetotext.ImageToLabel import ImageToLabel


class fsFileRecordView(ObjectEditorView):
    file_record_view: ft.Column

    def __init__(self, sel_object: object = None):
        super().__init__(sel_object)
        self.object: FileRecord = sel_object
        self.xml_source = """
        <root>
            <Column id="file_record_view">
                <Text value = "Settings"/>
                <Text value = "read:self.object.full_path" />
            </Column>
        </root>
        """
        self.parse_xml()
        self.file_record_view.expand_loose = True
        self.file_record_view.scroll = ft.ScrollMode.AUTO


class fsImageRecordViewProps(PropUser):
    caption_string: str = PropInfo()
    tags_string: str = PropInfo()


class fsImageRecordView(ObjectEditorView):
    image_ai_caption_text: ft.Text
    image_record_view: ft.Column
    all_tags_drop_down: ft.Dropdown

    def __init__(self, sel_object: object = None):
        super().__init__(sel_object)
        self.prop = fsImageRecordViewProps()
        self.object: FileRecord = sel_object
        self.xml_source = """
        <root>
            <Column id="image_record_view" >
                <Image src = "read:self.object.full_path"/>
                <Text value = "read:self.object.full_path" />
                <TextField id = "image_ai_caption_text" value="bind:caption_string"/>
                <Button text="Get caption" on_click="get_caption"/>
                <Row>
                    <DropDown id="all_tags_drop_down" label="all tags" />
                    <Button text="Add tag" />
                </Row>
                
                <TextField id="tags_text_field" value="bind:tags_string"/>
            </Column>
        </root>
        """

        self.parse_xml()

        self.image_record_view.expand_loose = True
        self.image_record_view.scroll = ft.ScrollMode.AUTO
        self.prop.dispatcher.tags_string.add_listener(self.on_tags_string_changed)

    def get_caption(self, event):
        val = ImageToLabel().get_label_from_path(self.object.full_path, "text_clip_2_1")

        self.prop.caption_string = val
        print(val)

    def on_tags_string_changed(self, *args, **kwargs):
        print("tags changed")


class obj_editor_view(Flet_view):
    toolbar_row: ft.Row
    center_column: ft.Column

    def __init__(self):
        super().__init__()

        self.gen_view_progress_bar = None
        self.gen_progress_message = None
        prog_man = ProgressManager.instance()
        prog_man.add_visualizer(self)
        self.expand = True
        self.close_event = lambda e: self.close_view()
        self.xml_source = """
        <root>
            <Row id="toolbar_row" >
                <MenuBar>
                    <SubmenuButton text="File">
                    </SubmenuButton>
                </MenuBar>
                <Button text="x" on_click="close_event"/>
            </Row>

            <Row expand="1">
                <Column id="left_column" expand="1">
                    <Text value = "Settings"/>
                </Column>

                <Column id="center_column" expand="3">
                </Column>
            </Row>
            <Row id="footer_row" wrap="True">
                <ProgressBar id="gen_view_progress_bar" value="0.0" width="300"/>
                <Text id="gen_progress_message" value="message"/>
            </Row>
        </root>
        """
        self.parse_xml()

        self.obj_editor = FtObjectEditor()
        self.obj_editor.add_view_template(fsImageRecordView,
                                          lambda obj: (isinstance(obj, FileRecord)) and obj.file_type == "image")
        self.obj_editor.add_view_template(fsImageRecordView,
                                          lambda obj: (isinstance(obj, FileRecord)))
        self.center_column.controls.append(self.obj_editor)
        if __name__ == "__main__":
            self.edit_object(
                FileRecord.add_file_record_from_path(
                    r"D:\image db\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"))

    def update_progress(self):
        prog_man = ProgressManager.instance()
        cur_prog = prog_man.progress / prog_man.max_progress
        self.gen_progress_message.value = prog_man.message
        self.gen_view_progress_bar.value = cur_prog
        self.update()

    def edit_object(self, obj):
        self.obj_editor.set_object(obj)


if __name__ == "__main__":
    view = obj_editor_view()
    app = FletGlueApp()
    app.set_view(view)

    app.run()
