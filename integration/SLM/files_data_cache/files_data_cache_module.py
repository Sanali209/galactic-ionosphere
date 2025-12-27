import os
import sys

from SLM.appGlue.core import Module, TypedConfigSection
from SLM.appGlue.core import Allocator
from SLM.files_data_cache.imageToLabel import ImageToTextCache


class FileDataManagerConfig(TypedConfigSection):
    path: str = os.path.dirname(sys.argv[0])


Allocator.config.register_section("fileDataManager", FileDataManagerConfig)


class FilesDataCacheModule(Module):
    def __init__(self):
        super().__init__("FilesDataCacheModule")

    def init(self):
        from SLM.files_data_cache.imagedatacache import ImageDataCacheManager
        from SLM.files_data_cache.md_5_backends.Md5PilContentBackend import Md5PilContentBackend
        from SLM.files_data_cache.thumbnail import ImageThumbCache

        img_data_cache_m = ImageDataCacheManager()
        Allocator.res.register(img_data_cache_m)

        img_data_cache_m.content_md5_provider.backends["jpg"] = Md5PilContentBackend()
        img_data_cache_m.content_md5_provider.backends["jpeg"] = Md5PilContentBackend()
        img_data_cache_m.content_md5_provider.backends["png"] = Md5PilContentBackend()

        Allocator.res.register(ImageToTextCache())
        Allocator.res.register(ImageThumbCache())
