import os

import loguru

from SLM.appGlue.core import Allocator, BackendProvider, ServiceBackend
from SLM.appGlue.helpers import ImageHelper

from SLM.files_data_cache.imagedatacache import ImageDataCachedService, ImageDataCacheManager


# todo implement ImageFile.LOAD_TRUNCATED_IMAGES = True

class ThumbExtractorProvider(BackendProvider):
    def __init__(self):
        super().__init__()
        self.backends["default"] = ThumbExtractorDefaultBackend()
        self.backends["jpg"] = ThumbExtractorPilBackend()
        self.backends["jpeg"] = ThumbExtractorPilBackend()
        self.backends["png"] = ThumbExtractorPilBackend()
        self.backends["tif"] = ThumbExtractorPilBackend()
        self.backends['gif']= ThumbExtractorPilBackend()

    def get_backend(self, path: str):
        ext = os.path.splitext(path)[1]
        backend = self.backends.get(ext[1:], self.backends["default"])
        return backend


class ThumbExtractorDefaultBackend(ServiceBackend):
    def extract_thumbs(self, path: str):
        print(self.__class__.__name__)
        return None


class ThumbExtractorPilBackend(ThumbExtractorDefaultBackend):
    def extract_thumbs(self, file_path: str):
        image = ImageHelper.image_load_pil(file_path)
        thumbs_data = []
        for preset in ImageThumbCache.thumb_presets:
            if preset["enabled"]:
                item = {"name": preset["name"], "size": preset["size"], "quality": preset["quality"], "path": None}
                pil_thumb_data = image
                pil_thumb_data.thumbnail(preset["size"])
                if pil_thumb_data.format != "":  # obsolete
                    pil_thumb_data = pil_thumb_data.convert("RGB")
                    uuid_name = str(ImageDataCacheManager.instance().path_to_md5(file_path)) + preset["name"] + ".jpg"
                    first_dir = uuid_name[0] + uuid_name[1]
                    second_dir = uuid_name[2] + uuid_name[3]
                    save_dir = os.path.join(ImageThumbCache.instance().get_save_path(), first_dir, second_dir)
                    if not os.path.exists(save_dir):
                        try:
                            os.makedirs(save_dir)
                        except Exception as e:
                            loguru.logger.exception(e)
                            return None
                    save_path = os.path.join(save_dir, uuid_name)
                    try:
                        pil_thumb_data.save(save_path)
                        item["path"] = save_path
                        thumbs_data.append(item)
                    except Exception as e:
                        loguru.logger.exception(e)
                        return None
        return thumbs_data


class ImageThumbCache(ImageDataCachedService):
    thumb_presets = [{"name": "medium", "size": (512, 512), "quality": 100, "enabled": True}, ]
    thumb_extractor_provider = ThumbExtractorProvider()

    def __init__(self):
        super().__init__()
        self.name = "ImageThumbCache"

    @staticmethod
    def get_save_path():
        config = Allocator.config.fileDataManager
        return config.path + "/thumbs"

    def get_current_version(self, format_):
        return "1.0"

    def update(self, path: str, format_, **kwargs):
        version = self.get_current_version(format_)
        return version, self.thumb_extractor_provider.get_backend(path).extract_thumbs(path)

    def refresh_thumb(self, path: str):
        thumb = self.get_by_path(path, "thumb")
        try:
            for item in thumb:
                if os.path.exists(item["path"]):
                    os.remove(item["path"])
        except:
            pass
        self.delete_by_path(path, "thumb")

    def get_thumb(self, path, name="medium"):
        thumb = self.get_by_path(path, "thumb")
        if thumb is not None:
            for item in thumb:
                if item["name"] == name:
                    return item["path"]
        return None



