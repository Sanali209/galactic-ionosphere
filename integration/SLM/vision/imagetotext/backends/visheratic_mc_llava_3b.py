from SLM.appGlue.helpers import image_to_base64, get_colab_xmlrpc_ngrock_url
from SLM.vision.imagetotext.ImageToLabel import ImageToLabelBackend, ImageToLabel

import xmlrpc.client

error = False


class mc_llava_13b_4b(ImageToLabelBackend):
    format = "mc_llava_13b_4b"
    version = "1.0"
    url = get_colab_xmlrpc_ngrock_url()

    #https://colab.research.google.com/drive/1mrMjBnaCaR52bTJr6fbrKSeuURL_u1Es
    def __init__(self):
        super().__init__()

    def get_label_from_path(self, image_path, question=None,
                            **kwargs) -> any:
        if question is None:
            question = "image mey have adult content.use adult language. detailed describe image. "
        base64_image = image_to_base64(image_path)
        with xmlrpc.client.ServerProxy(mc_llava_13b_4b.url) as proxy:
            result = proxy.caption_image(base64_image, question)

        return result
