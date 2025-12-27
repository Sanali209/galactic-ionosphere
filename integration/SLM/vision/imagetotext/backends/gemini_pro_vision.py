import os

from PIL import Image

from langchain_core.messages import HumanMessage

from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotext.ImageToLabel import ImageToLabelBackend, ImageToLabel

try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        HarmBlockThreshold,
        HarmCategory,
    )

    error = False
except Exception as e:
    error = True

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyB2n4fQmeKYpGId5qdWdClp1wHEnz0vQic'


class gemini_pro_vision(ImageToLabelBackend):
    format = "gemini_pro_vision"

    def __init__(self):
        super().__init__()
        # change models by need (gemini-pro, gemini-pro-vision)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", maxOutputTokens=2048, safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, }
                                          )

    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        pil_image = PILPool.get_pil_image(image_path)
        messages = [HumanMessage(content=[
            {"type": "text",
             "text": "What's in this image?"},
            {"type": "image_url",
             "image_url": pil_image},
        ])]

        result = self.llm.invoke(messages)
        return result


def load_module():
    if not error:
        ImageToLabel.all_backends[gemini_pro_vision.format] = gemini_pro_vision


load_module()
