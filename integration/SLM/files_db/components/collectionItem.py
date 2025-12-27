from typing import Dict

from SLM.appGlue.core import Event
from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo


class CollectionRecord(MongoRecordWrapper):
    itemTypeMap: Dict[str, type] = {}
    itemType: str = FieldPropInfo('item_type', str, 'CollectionRecord')
    favorite: bool = FieldPropInfo('favorite', bool, False)
    hidden: bool = FieldPropInfo('hidden', bool, False)
    rating: int = FieldPropInfo('rating', int, 0)
    document_content: str = FieldPropInfo('document_content', str, '')
    full_text_search: str = FieldPropInfo('full_text_search', str, '')
    title: str = FieldPropInfo('title', str, '')
    description: str = FieldPropInfo('description', str, '')
    notes: str = FieldPropInfo('notes', str, '')
    ai_expertise: [] = FieldPropInfo('ai_expertise', str, [])
    file_content_md5: str = FieldPropInfo('file_content_md5', str, '')
    metadata_dirty: bool = FieldPropInfo('metadata_dirty', bool, False)
    name: str = FieldPropInfo('name', str, None)
    local_path: str = FieldPropInfo('local_path', str, None)
    url_source: str = FieldPropInfo('source', str, None)

    @classmethod
    def get_record_wrapper(cls, record_id):
        record_data = cls.find_one({'_id': record_id})
        wrapper = cls.itemTypeMap[record_data.itemType](record_id)
        return wrapper

    def get_thumb(self, size=None):
        pass

    @classmethod
    def get_record_wrapper2(cls, record_id, record_type):
        """more performant version of get_record_wrapper"""
        wrapper = cls.itemTypeMap[record_type](record_id)
        return wrapper

    @classmethod
    def find_one(cls, query):
        propInfo = cls.__dict__["itemType"]
        query["item_type"] = propInfo.default
        res = cls.collection().find_one(query)
        if res is None: return None
        return cls(res["_id"])

    @classmethod
    def find(cls, query, sort_query=None):
        prop_info = cls.__dict__["itemType"]
        query["item_type"] = prop_info.default
        if sort_query is None:
            res = cls.collection().find(query)
        else:
            res = cls.collection().find(query).sort(sort_query)
        return [cls(item["_id"]) for item in res]
