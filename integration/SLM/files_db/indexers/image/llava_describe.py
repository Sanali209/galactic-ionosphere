
from SLM.files_db.indexers.image.content_md5 import files_db_indexer

from SLM.chains.chains_main import DictFieldMergeChainFunction, DictFormatterChainFunction
from SLM.groupcontext import group
from SLM.indexerpyiplain.idexpyiplain import ItemIndexer
from SLM.metadata.MDManager.mdmanager import MDManager


# todo :tags black list
# todo: tags sinonims


class ImageLLavaDescribe(files_db_indexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fieldName = "ImageLLavaDescribe"

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord
        file_item = FileRecord(item["_id"])

        result = file_item.get_ai_expertise("image-text", "mc_llava_13b_4b", question="detailed describe image")

        file_item.description = result['data']

        parent_indexer.shared_data["item_indexed"] = True

        self.mark_as_indexed(item, parent_indexer)



