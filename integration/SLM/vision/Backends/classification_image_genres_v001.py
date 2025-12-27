from SLM.vision.LLMBackend import LLMBackend
from PIL import Image

class classification_image_genres_v001(LLMBackend):
    name = "classification_image_genres_v001"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_image_classification(self, Image: Image,single_label: bool = True) ->  list:
        if single_label==False:
            raise NotImplementedError
        raise NotImplementedError