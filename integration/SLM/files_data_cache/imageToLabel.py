from SLM.appGlue.core import Allocator
from SLM.files_data_cache.imagedatacache import ImageDataCachedService
from SLM.vision.imagetotext.ImageToLabel import ImageToLabel


class ImageToTextCache(ImageDataCachedService):
    @staticmethod
    def formats():
        return list(ImageToLabel.all_backends.keys())

    def __init__(self):
        super().__init__()
        self.name = "ImageToText"

    def get_current_version(self, format_):
        return ImageToLabel.get_backend_version(format_)

    def update(self, path: str, format_, **kwargs):
        version = ImageToLabel.get_backend_version(format_)
        return version, ImageToLabel.get_label_from_path(path, format_, **kwargs)

