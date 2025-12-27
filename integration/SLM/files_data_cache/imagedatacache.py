import hashlib
import os

from typing import Any

from diskcache import Index


from SLM.appGlue.core import Service, ServiceBackend, BackendProvider


class Md5DefaultContentBackend(ServiceBackend):
    def get_md5(self, path: str):
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error calculating MD5 for {path}: {e}")
            return None


class Md5ContentProvider(BackendProvider):
    def __init__(self):
        super().__init__()
        self.backends["default"] = Md5DefaultContentBackend()

    def get_backend(self, path: str):
        try:
            ext: str = os.path.splitext(path)[1]
            ext = ext.lower()
        except Exception as e:
            return self.backends["default"]
        backend = self.backends.get(ext[1:], self.backends["default"])
        return backend


class ImageDataCacheManager(Service):
    """Class for caching md5 of image content.
    """
    content_md5_provider = Md5ContentProvider()

    def __init__(self):
        super().__init__()
        self.pats_md5: Index | None = None

    def init(self, config):
        self.pats_md5 = Index(os.path.join(config.fileDataManager.path, 'pats_md5'))

    def path_to_md5(self, path: str):
        val = self.pats_md5.get(path, default=None)
        if val is None:
            val = self.content_md5_provider.get_backend(path).get_md5(path)
            if val is None:
                if path in self.pats_md5:
                    del self.pats_md5[path]
                return None
            self.pats_md5[path] = val
        return val

    def set_path_to_md5(self, path: str, md5: str):
        self.pats_md5[path] = md5


class ImageDataCachedService(Service):
    def __init__(self):
        super().__init__()
        self.name = "ImageDetect_cached"
        self.path = ""
        self.formats_caches = {}

    def init(self, config):
        self.path = os.path.join(config.fileDataManager.path, self.name)

    def is_need_update(self, value, formath_):
        if value is None:
            return True
        version, cached_value = value
        if version != self.get_current_version(formath_):
            return True
        return False

    def get_by_path(self, path: str, format_: str, **kwargs):
        if format_ not in self.formats_caches.keys():
            self.formats_caches[format_] = Index(os.path.join(self.path, format_))
        imdatamanager = ImageDataCacheManager.instance()
        md5 = imdatamanager.path_to_md5(path)
        kwarg_str = ""
        kwargs_md5 = ""
        for key in kwargs.keys():
            kwarg_str += str(kwargs[key])
        if kwarg_str != "":
            kwargs_md5 = str(hashlib.md5(kwarg_str.encode()).hexdigest())

        if md5 is None or kwargs_md5:
            return None

        value = self.formats_caches[format_].get(md5 + kwargs_md5, default=None)

        if self.is_need_update(value, format_):
            value = self.update(path, format_, **kwargs)
            self.formats_caches[format_][md5 + kwargs_md5] = value
        return value[1]

    def get_by_md5(self, md5: str, format_: str, **kwargs):
        if format_ not in self.formats_caches.keys():
            self.formats_caches[format_] = Index(os.path.join(self.path, format_))
        kwarg_str = ""
        kwargs_md5 = ""
        for key in kwargs.keys():
            kwarg_str += str(kwargs[key])
        if kwarg_str != "":
            kwargs_md5 = str(hashlib.md5(kwarg_str.encode()).hexdigest())

        if md5 is None or kwargs_md5:
            return None

        value = self.formats_caches[format_].get(md5 + kwargs_md5, default=None)

        if value is None:
            return None

        return value[1]

    def set_by_md5(self, md5: str, format_: str, value, **kwargs):
        if format_ not in self.formats_caches.keys():
            self.formats_caches[format_] = Index(os.path.join(self.path, format_))
        kwarg_str = ""
        kwargs_md5 = ""
        for key in kwargs.keys():
            kwarg_str += str(kwargs[key])
        if kwarg_str != "":
            kwargs_md5 = str(hashlib.md5(kwarg_str.encode()).hexdigest())
        self.formats_caches[format_][md5 + kwargs_md5] = value

    def get_current_version(self, format_):
        raise NotImplementedError("get_current_version method must be implemented")

    def delete_by_path(self, path: str,format_: str,**kwargs):
        if format_ not in self.formats_caches.keys():
            self.formats_caches[format_] = Index(os.path.join(self.path, format_))
        kwarg_str = ""
        kwargs_md5 = ""
        for key in kwargs.keys():
            kwarg_str += str(kwargs[key])
        if kwarg_str != "":
            kwargs_md5 = str(hashlib.md5(kwarg_str.encode()).hexdigest())
        md5 = ImageDataCacheManager.instance().path_to_md5(path)
        self.formats_caches[format_].pop(md5 + kwargs_md5)

    def update(self, path: str, format_, **kwargs) -> tuple[str, Any]:
        """
        return tuple (version, value)
        :param format_:
        :param path:
        :param kwargs:
        :return: tuple (version, value)
        """
        version = self.get_current_version(format_)
        return version, None  # return value for cache
