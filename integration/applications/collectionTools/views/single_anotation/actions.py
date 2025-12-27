import os

from tqdm import tqdm

from SLM.actions import ActionManager, AppAction
from SLM.appGlue.DesignPaterns import allocator
from SLM.appGlue.glue_app import GlueApp
from SLM.flet.flet_ext import flet_dialog_alert

# todo integrate this in collection tools app
@ActionManager().register()
class AppActionMoveToDirectoryByAnnotation(AppAction):
    def __init__(self):
        super().__init__(name="move to directory by annotation", description="move to directory by annotation")

    def run(self, *args, **kwargs):
        app = allocator.Allocator.get_instance(GlueApp)
        path = args[0]
        if not os.path.exists(path):
            dialog = flet_dialog_alert(title="Error", content="path not exists")
            dialog.show()
            return
        job_name = args[1]

        query = select(ImageSideCard).where(col(ImageSideCard.image_path).like(f"{path}%"))
        result = SideCardDB().session.exec(query).all()
        for image in tqdm(result):
            annotations = Image_Prediction_Helper.get_predictions_by_name(image, job_name)
            if len(annotations) == 0:
                continue
            annotation = annotations[0]
            print(f"{image.image_path} -> {annotation.label}")
            dir_name = os.path.dirname(image.image_path) + "\\" + annotation.label
            sidecard_helper.move_image_to_path(image, dir_name)
