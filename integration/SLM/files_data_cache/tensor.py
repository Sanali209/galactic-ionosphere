import os

from diskcache import Index
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.files_data_cache.imagedatacache import ImageDataCacheManager


# todo: realise embed function functionality
class Embeddings_cache:
    def __init__(self, load_caches: list[str]):
        """

        @param load_caches: list of formats to load. available formats: 'modilenetv3','alexNet'
        """
        self.formats_caches = {}
        config = Allocator.config.fileDataManager
        if not os.path.exists(os.path.join(config.path, "embeddings")):
            os.makedirs(os.path.join(config.path, "embeddings"))
        for format in load_caches:
            self.formats_caches[format] = Index(os.path.join(config.path, "embeddings", format),cull_limit=0, size_limit=100 * 1024 * 1024 * 1024)
        self.image_data_manager = ImageDataCacheManager.instance()

    def update_from_external(self, path: str, t_format: str):
        ext_index = Index(path)
        for key in ext_index.keys():
            self.formats_caches[t_format][key] = ext_index[key]

    def get_by_path(self, path: str, t_format: str):
        md5 = self.image_data_manager.path_to_md5(path)
        value = self.formats_caches[t_format].get(md5, default=None)
        if self.is_need_update(value):
            value = self.update(path, t_format)
            if value is None:
                if md5 in self.formats_caches[t_format]:
                    del self.formats_caches[t_format][md5]
                return None
            self.formats_caches[t_format][md5] = value
        return value

    def is_need_update(self, value):
        if value is None:
            return True
        return False

    def update(self, path: str, t_format):
        from SLM.vision.imagetotensor.CNN_Encoding import ImageToCNNTensor
        return ImageToCNNTensor.get_tensor_from_path(path, t_format)

    def import_from_other_cache(self, path, t_format: str):
        disck_cache = Index(path)
        for key, value in tqdm(disck_cache.items()):
            self.formats_caches[t_format][key] = value