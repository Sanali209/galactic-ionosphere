from SLM.appGlue.core import ServiceBackend


class Md5PilContentBackend(ServiceBackend):
    def get_md5(self, path: str):
        from SLM.appGlue.helpers import ImageHelper
        try:
            image = ImageHelper.image_load_pil(path)
        except Exception as e:
            return None
        return ImageHelper.content_md5(image)
