from pathlib import Path

from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotext.ImageToLabel import ImageToLabelBackend, ImageToLabel
from gradio_client import Client, handle_file

from SLM.vision.imagetotext.ddtest import image_to_deepdanbooru_tags


class DeepDanbury(ImageToLabelBackend):
    format = "DeepDanbury"

    def __init__(self):
        super().__init__()

    def get_label_from_path(self, image_path, *kwargs) -> any:
        image = PILPool.get_pil_image(image_path,False)
        res = image_to_deepdanbooru_tags(image, 0.5, False, False, False, True)
        return res[0]


ImageToLabel.all_backends[DeepDanbury.format] = DeepDanbury
