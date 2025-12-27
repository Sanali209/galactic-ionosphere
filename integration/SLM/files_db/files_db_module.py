from SLM.appGlue.core import Module, TypedConfigSection, Allocator
from SLM.files_data_cache.tensor import Embeddings_cache
from SLM.files_db.components.File_record_wraper import FileRecord

from SLM.files_db.object_recognition.object_recognition import DetectionObjectClass, Detection, Recognized_object
from SLM.vision.imagetotensor.backends.BLIP import CNN_Encoder_BLIP
from SLM.vision.imagetotensor.backends.DINO import CNN_Encoder_DINO
from SLM.vision.imagetotensor.backends.clip_vit_dirml import CNN_Encoder_CLIP_DML
from SLM.vision.imagetotensor.backends.inceptionV3 import CNN_Encoder_InceptionV3
from SLM.vision.imagetotensor.backends.inception_resnet_v2 import CNN_Encoder_InceptionResNetV2
from SLM.vision.imagetotensor.backends.mobile_net_v3 import CNN_encoder_ModileNetv3_Small
from SLM.vision.imagetotensor.backends.resnet import CNN_Encoder_ResNet50
from SLM.vision.imagetotensor.backends.resnetinceptionfacenet512 import CNN_Encoder_FaceNet
from SLM.vision.imagetotensor.custom.custom_emb import CNN_Encoder_custom
from SLM.vision.imagetotensor.custom_mobile_net.custom_mobv2_emb import CNN_Encoder_mv2_custom
from SLM.vision.vector_fuse import EmbeddingFusion

emb_cache = None


def vectorize_face_FaceNet(face: Detection):
    #todo improve by using tumb api
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_FaceNet.format)
    return vector


def vectorize_face_ResNet50(face: Detection):
    #todo improve by using tumb api
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_ResNet50.format)
    return vector


def vectorize_FileRecord_ResNet50(file: FileRecord):
    #todo improve by using tumb api
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_ResNet50.format)
    return vector


def vectorize_face_InceptionResNetV2(face: Detection):
    #todo improve by using tumb api
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_InceptionResNetV2.format)
    return vector


def vectorize_FileRecord_InceptionResNetV2(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_InceptionResNetV2.format)
    return vector


def vectorize_face_InceptionV3(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_InceptionV3.format)
    return vector


def vectorize_FileRecord_InceptionV3(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_InceptionV3.format)
    return vector


def vectorize_face_ModileNetv3_Small(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_encoder_ModileNetv3_Small.format)
    return vector


def vectorize_FileRecord_ModileNetv3_Small(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_encoder_ModileNetv3_Small.format)
    return vector


def vectorize_face_CLIP_DML(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_CLIP_DML.format)
    return vector


def vectorize_FileRecord_CLIP_DML(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_CLIP_DML.format)
    return vector


def vectorize_face_DINO(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_DINO.format)
    return vector


def vectorize_FileRecord_DINO(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_DINO.format)
    return vector


def vectorize_face_BLIP(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_BLIP.format)
    return vector


def vectorize_FileRecord_BLIP(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_BLIP.format)
    return vector


def vectorize_face_custom(face: Detection):
    path = face.obj_image_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_custom.format)
    return vector


def vectorize_FileRecord_custom(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_custom.format)
    return vector


def vectorize_FileRecord_mv2_custom(file: FileRecord):
    path = file.full_path
    vector = emb_cache.get_by_path(path, CNN_Encoder_mv2_custom.format)
    return vector

def vectorize_fileRecord_fuse_MCBD(file: FileRecord):
    """
    Функция для векторизации FileRecord с использованием нескольких моделей.
    Возвращает объединённый вектор.
    """
    fusion = EmbeddingFusion(target_dim=768)
    fusion.add_embedding("mobile_net_v3", vectorize_FileRecord_ModileNetv3_Small(file), weight=0.1)
    fusion.add_embedding("clip", vectorize_FileRecord_CLIP_DML(file), weight=0.4)
    fusion.add_embedding("blip", vectorize_FileRecord_BLIP(file), weight=0.3)
    fusion.add_embedding("dino", vectorize_FileRecord_DINO(file), weight=0.2)
    fused_vext = fusion.fuse()
    return fused_vext




class IndexerConfig(TypedConfigSection):
    detectFaces: bool = True


Allocator.config.register_section("Indexer", IndexerConfig)


class FilesDBModule(Module):
    def __init__(self):
        super().__init__("FilesDBModule")

    def load(self):
        from SLM.mongoext.MongoClientEXT_f import MongoClientExt
        from SLM.mongoext.wraper import MongoRecordWrapper
        from SLM.files_db.components.collectionItem import CollectionRecord
        from SLM.files_db.components.File_record_wraper import FileRecord
        from SLM.files_db.components.fs_tag import TagRecord
        from SLM.files_db.components.relations.relation import RelationRecord
        from SLM.files_db.annotation_tool.annotation import AnnotationRecord, AnnotationJob
        global emb_cache
        emb_cache = Embeddings_cache([CNN_Encoder_FaceNet.format, CNN_Encoder_BLIP.format, CNN_Encoder_DINO.format,
                                      CNN_Encoder_InceptionResNetV2.format, CNN_Encoder_InceptionV3.format,
                                      CNN_encoder_ModileNetv3_Small.format, CNN_Encoder_ResNet50.format,
                                      CNN_Encoder_CLIP_DML.format, CNN_Encoder_custom.format,
                                      CNN_Encoder_mv2_custom.format])

        mongo_client = MongoClientExt.instance()

        MongoRecordWrapper.client = mongo_client

        mongo_client.register_collection("collection_records", CollectionRecord)

        mongo_client.register_collection("collection_records", FileRecord)
        CollectionRecord.itemTypeMap['FileRecord'] = FileRecord
        FileRecord.create_index("local_path|name", {'local_path': 1, 'name': 1})
        FileRecord.create_index("file_cont_md5I", {'file_content_md5': 1})
        FileRecord.create_index("text_search", {'name': 'text', 'document_content': 'text',
                                                'description': 'text', 'notes': 'text',
                                                'title': 'text', 'tags': 'text', 'local_path': 'text'})
        FileRecord.create_index("local_path", {'local_path': 1})

        mongo_client.register_collection("tags_records", TagRecord)
        TagRecord.create_index("fullName", {'fullName': 1})
        TagRecord.create_index("parent_tag", {'parent_tag': 1})

        mongo_client.register_collection("relation_records", RelationRecord)
        RelationRecord.create_index("from_id", {'from_id': 1})
        RelationRecord.create_index("from_id|distance|type", {'from_id': 1, 'distance': 1, 'type': 1})
        RelationRecord.create_index("type|sub_type|distance", {'type': 1, 'sub_type': 1, 'distance': 1})
        # TODO add deleting anotation acordet to file
        #CollectionRecord.onDeleteGlobal += RelationRecord.delete_all_relations

        mongo_client.register_collection("object_class_dict", DetectionObjectClass)
        DetectionObjectClass.create_index("name", {'name': 1})

        CollectionRecord.itemTypeMap['Detection'] = Detection
        mongo_client.register_collection("collection_records", Detection)

        CollectionRecord.itemTypeMap['Recognized_object'] = Recognized_object
        mongo_client.register_collection("collection_records", Recognized_object)
        Detection.create_index("parent_image_id", {'parent_image_id': 1})
        #FileRecord.onDeleteGlobal += Detection.del_detection

        service = Allocator.get_instance(MongoClientExt)
        service.register_collection("annotation_records", AnnotationRecord)
        service.register_collection("annotation_job_records", AnnotationJob)
        # check 'not_annotated' not in indexes:
        AnnotationJob.create_index('not_annotated_id_ind', {'_id': 1, 'not_annotated': 1})
        AnnotationRecord.create_index('parent_id_file_id1', {'parent_id': 1, 'file_id': 1})
        AnnotationRecord.create_index('file_id1', {'file_id': 1})
        AnnotationRecord.create_index('parent_id1', {'parent_id': 1})
        AnnotationRecord.create_index('parent_value', {'parent_id': 1, 'value': 1})
