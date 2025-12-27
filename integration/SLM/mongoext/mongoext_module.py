from SLM.appGlue.core import Module, Allocator
from SLM.mongoext.wraper import MongoRecordWrapper


class MongoExtModule(Module):
    def __init__(self):
        super().__init__("MongoExtModule")

    def init(self):
        from SLM.mongoext.MongoClientEXT_f import MongoClientExt
        Allocator.res.register(MongoClientExt())
        MongoRecordWrapper.client = MongoClientExt.instance()
