import hashlib
import os
from abc import ABC

from PIL import Image

from SLM import Allocator
from SLM.appGlue.core import ServiceBackend, BackendProvider
from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
from SLM.files_data_cache.pool import PILPool
from diskcache import Index

from SLM.files_data_cache.thumbnail import ImageThumbCache


class LLMBackend(ServiceBackend):
    format = "LLMBackend"
    vector_size = 0
    version = "1.0.0"
    threshold_default = {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}
    image_datacache_manager = None
    image_tumb_manager: ImageThumbCache = None

    def __init__(self, *args, **kwargs):
        self.index: Index = None

    def get_curent_version(self):
        return self.version

    def load(self):
        config = Allocator.config.fileDataManager
        if not os.path.exists(os.path.join(config.path, "llm_data-cache")):
            os.makedirs(os.path.join(config.path, "llm_data"))
        self.index = Index(os.path.join(config.path, "embeddings", format), cull_limit=0,
                           size_limit=100 * 1024 * 1024 * 1024)
        cached_version = self.index.get("version", self.get_curent_version())
        curent_version = self.get_curent_version()
        if cached_version != curent_version:
            self.index.clear()
            self.index["version"] = curent_version
        self.image_datacache_manager = ImageDataCacheManager.instance()
        self.image_tumb_manager = ImageThumbCache()

    def get_PIL_image_tumb_and_md5(self, image_path: str, thumb_name="medium"):
        image_thumb_path = self.image_tumb_manager.get_thumb(image_path, thumb_name)
        image = PILPool.get_pil_image(image_path, copy=False)
        md5 = self.image_datacache_manager.path_to_md5(image_path)
        return image, md5

    def kwargs_to_key(self, **kwargs):
        kwarg_str = ""
        kwargs_md5 = ""
        for key in kwargs.keys():
            kwarg_str += str(key) + str(kwargs[key])
        if kwarg_str != "":
            kwargs_md5 = str(hashlib.md5(kwarg_str.encode()).hexdigest())

    def set_cached(self, key: str, data):
        self.index[key] = data

    def get_cached(self, key):
        return self.index.get(key)

    def cache(self, key, data=None, **kwargs):
        key2 = self.kwargs_to_key(kwargs)
        if data == None:
            cached_data = self.get_cached(key + key2)
            return cached_data
        else:
            self.set_cached(key + key2)

    def get_image_tensor(self, image: Image, ):
        raise NotImplementedError

    def get_text_tensor(self, text: str):
        raise NotImplementedError

    def get_text_mach(self, Image: Image, texts: list[str]) -> (str, list[float]):
        raise NotImplementedError

    def get_image_classification(self, Image: Image, single_label: bool = True) -> list:
        raise NotImplementedError

    def get_image_description(self, image: Image) -> str:
        raise NotImplementedError

    def get_image_qa(self, image: Image):
        raise NotImplementedError

    def get_image_detection(self, image: Image) -> list:
        raise NotImplementedError

    def get_image_data(self, image: Image):
        raise NotImplementedError

    def get_image_data_from_path(self, image_path: str):
        raise NotImplementedError

    def text_inference(self, question: str) -> str:
        raise NotImplementedError

    def multimodal_inference(self, image: Image, question: str) -> str:
        raise NotImplementedError

    def generate_image(self, text: str) -> Image:
        raise NotImplementedError


class LLMBackendProvider(BackendProvider):
    name = "LLMBackendProvider"
