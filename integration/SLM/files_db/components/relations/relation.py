from SLM.files_db.components.collectionItem import CollectionRecord
from SLM.mongoext.wraper import MongoRecordWrapper, FieldPropInfo


class RelationRecord(MongoRecordWrapper):
    type: str = FieldPropInfo('type', str, '')
    sub_type: str = FieldPropInfo('sub_type', str, '')
    from_id: str = FieldPropInfo('from_id', str, '')
    to_id: str = FieldPropInfo('to_id', str, '')
    flags:list = FieldPropInfo('flags', list, [])
    data: dict = FieldPropInfo('data', dict, {})

    @classmethod
    def get_or_create(cls, from_: CollectionRecord, to_: CollectionRecord, type: str = "None", **kwargs):
        kwargs["from_id"] = from_._id
        kwargs["to_id"] = to_._id
        kwargs["type"] = type
        return cls._get_or_create(**kwargs)

    @classmethod
    def set_relation(cls, from_: CollectionRecord, to_: CollectionRecord, type_: str = "None"):
        return cls.get_or_create(from_, to_, type_)

    @classmethod
    def get_outgoing_relations(cls, from_CollectionRecord, type_: str = "None"):
        return cls.find({'from_id': from_CollectionRecord._id, 'type': type_})

    @classmethod
    def is_exist(cls, from_: CollectionRecord, to_: CollectionRecord, type_: str = "None"):
        return cls.find_one({'from_id': from_._id, 'to_id': to_._id, 'type': type_}) is not None

    @classmethod
    def delete_all_relations(cls, obj):
        if issubclass(obj.__class__, CollectionRecord):
            RelationRecord.collection().delete_many({'$or': [{'from_id': obj._id}, {'to_id': obj._id}]})




