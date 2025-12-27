import copy
import fnmatch
import os

from SLM.appGlue.core import Service, Allocator
from SLM.files_db.indexers.image.deepdanboru import DeepDunBoru, SmilingWolfTagger
from SLM.files_db.indexers.image.face_detection import FaceDetector
from SLM.files_db.indexers.image.llava_describe import ImageLLavaDescribe
from SLM.files_db.indexers.image.metadata_read import Image_MetadataRead
from SLM.files_db.indexers.image.tags_from_name import ImageTagsFromName

from SLM.indexerpyiplain.idexpyiplain import ItemIndexer


class FileTypeRouter(Service):
    def __init__(self):
        super().__init__()
        self.file_schemes = [ImageJPG.instance(), ImagePNG.instance()]
        self.default = File()

    def get_type_by_path(self, path) -> 'FileScheme':
        extension = os.path.splitext(path)[1]
        for scheme in self.file_schemes:
            for pattern in scheme.mach_patterns:
                if fnmatch.fnmatch(pattern, extension):
                    return scheme
        return self.default

    def get_type_by_name(self, name: str) -> 'FileScheme':
        for scheme in self.file_schemes:
            if scheme.name.lower() == name.lower():
                return scheme
        return self.default


class FileScheme(Service):
    def __init__(self):
        super().__init__()
        self.name = ""
        self.mach_patterns = [""]
        self.type = ""
        self.content = ""
        self.indexer = None
        self.attachments = {}

    def get_att(self, key, copy_=False):
        val = self.attachments.get(key, None)
        if copy_:
            return copy.deepcopy(val)
        return self.attachments.get(key, None)


class File(FileScheme):
    def __init__(self):
        super().__init__()
        self.name = "File"
        self.ext = ["*"]
        self.content = "Unknown"


class ImageJPG(FileScheme):
    name = "Image:JPG"

    def __init__(self):
        super().__init__()
        self.name = "Image:JPG"
        self.mach_patterns = [".jpg", ".jpeg"]
        self.content = "Image"

    def init(self, config):
        image_indexer = ItemIndexer()
        meta_reader = Image_MetadataRead()

        meta_reader.enabled = True
        face_d = FaceDetector()
        face_d.enabled = False
        tags_from_name = ImageTagsFromName()
        tags_from_name.enabled = False
        llava_describe = ImageLLavaDescribe()
        llava_describe.enabled = False
        dd_tager = DeepDunBoru()
        dd_tager.enabled = False
        SW_tagger = SmilingWolfTagger()
        SW_tagger.enabled = True

        (
                image_indexer | face_d | meta_reader | tags_from_name | llava_describe | dd_tager | SW_tagger)
        self.attachments["base_indexer"] = image_indexer


class ImagePNG(FileScheme):
    name = "Image:PNG"

    def __init__(self):
        super().__init__()
        self.name = "Image:PNG"
        self.mach_patterns = [".png"]
        self.content = "Image"

    def init(self, config):
        image_indexer = ItemIndexer()
        meta_reader = Image_MetadataRead()

        meta_reader.enabled = True
        face_d = FaceDetector()
        face_d.enabled = False
        tags_from_name = ImageTagsFromName()
        tags_from_name.enabled = False
        llava_describe = ImageLLavaDescribe()
        llava_describe.enabled = False
        dd_tager = DeepDunBoru()
        dd_tager.enabled = False
        SW_tagger = SmilingWolfTagger()
        SW_tagger.enabled = True

        (
                image_indexer | face_d | meta_reader | tags_from_name | llava_describe | dd_tager | SW_tagger)
        self.attachments["base_indexer"] = image_indexer


Allocator.res.register(ImageJPG())
Allocator.res.register(ImagePNG())
Allocator.res.register(FileTypeRouter())
