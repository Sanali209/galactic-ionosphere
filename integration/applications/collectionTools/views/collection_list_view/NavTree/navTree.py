from SLM.appGlue.DesignPaterns import allocator
from SLM.appGlue.DesignPaterns.allocator import Service
from SLM.files_db.object_recognition.object_recognition import DetectionObjectClass, Recognized_object
from SLM.files_db.components.fs_tag import TagRecord
from SLM.flet.hierarhicallist import ftTreeView
from applications.collectionTools.views.collection_list_view.NavTree.item_models_view import FsRootItemNode, \
    FsFolderInfoItemNode, RecordTypeRootItemNode, RecordTypeModelItemNode
from applications.collectionTools.views.collection_list_view.NavTree.items_models import FsRoot, FsFolderInfo, \
    RecordTypeRoot, RecordTypeModel
from applications.collectionTools.views.collection_list_view.NavTree.object_recognition.object_recognition_nav_tree import \
    RecognizedRoot, RecognizedRootItemNode, RecognizedObjectItemNode, RecognizedObjectClassItemNode
from applications.collectionTools.views.collection_list_view.NavTree.tags.tags import TagRoot, TagRootItemNode, \
    TagItemNode


class NavTreeSystem(Service):
    def __init__(self):
        super().__init__()
        self.naw_tree = ftTreeView()
        self.root_nodes = [FsRoot(), TagRoot(), RecordTypeRoot(), RecognizedRoot()]
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(FsRoot, FsRootItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(FsFolderInfo, FsFolderInfoItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(TagRoot, TagRootItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(TagRecord, TagItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(RecordTypeRoot, RecordTypeRootItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(RecordTypeModel, RecordTypeModelItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(RecognizedRoot, RecognizedRootItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(DetectionObjectClass,
                                                                     RecognizedObjectClassItemNode)
        self.naw_tree.viewTemplate.itemTemplateSelector.add_template(Recognized_object, RecognizedObjectItemNode)
        for item in self.root_nodes:
            self.naw_tree.add_item(item)


allocator.Allocator().register(NavTreeSystem, NavTreeSystem())
