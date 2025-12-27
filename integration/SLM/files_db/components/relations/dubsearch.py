import copy

from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from pymongo import InsertOne
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.files_db.components.File_record_wraper import FileRecord, get_file_record_by_folder
from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.files_db_module import vectorize_face_FaceNet, vectorize_face_ResNet50, vectorize_face_InceptionV3, \
    vectorize_face_ModileNetv3_Small, vectorize_face_CLIP_DML, vectorize_face_DINO, vectorize_face_BLIP, \
    vectorize_FileRecord_ModileNetv3_Small, vectorize_FileRecord_InceptionResNetV2, vectorize_FileRecord_InceptionV3, \
    vectorize_FileRecord_CLIP_DML, vectorize_FileRecord_DINO, vectorize_FileRecord_BLIP, vectorize_FileRecord_custom, \
    vectorize_FileRecord_mv2_custom, vectorize_FileRecord_ResNet50, vectorize_fileRecord_fuse_MCBD
from SLM.files_db.object_recognition.object_recognition import Detection
from SLM.iterable.bach_builder import BatchBuilder

from SLM.vector_db.vector_db import VectorDB, SearchScopeList, ResultGroup, ResultItem
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

search_batches = []


class Image_rel_flags(Enum):
    is_wrong = "is_wrong"
    is_near_similar = "is_near_similar"
    is_some_person = "is_some_person"


class ImageRelationHelper:
    type_name = "image_relation"

    def is_exist(self, from_id, to_id):
        query = {"$or": [{'from_id': from_id, 'to_id': to_id, 'type': ImageRelationHelper.type_name},
                         {'from_id': to_id, 'to_id': from_id, 'type': ImageRelationHelper.type_name}]}
        return RelationRecord.find_one(query) is not None

    def set_backend_data(self, relation: RelationRecord, backend_data: dict):
        dict_to_set = relation.data
        for key, value in backend_data.items():
            dict_to_set[key] = value
        relation.data = dict_to_set

    def create_new_relation(self, from_id, to_id, backend_data=None):
        if self.is_exist(from_id, to_id):
            return None
        if backend_data is not None:
            backend_data = copy.deepcopy(backend_data)
        else:
            backend_data = {}
        record = InsertOne({'from_id': from_id, 'to_id': to_id, 'type': ImageRelationHelper.type_name,
                            'data': backend_data})
        return record


class DubRelation:
    rel_type_name = "similar_obj_search"
    prefs_dict = {'face': [{'preset_name': 'face_vgg', 'vector_size': CNN_Encoder_FaceNet.vector_size,
                            'vectorize_func': vectorize_face_FaceNet, 'metric': 'angular'}],
                  'person': [{'preset_name': 'person_ResNet', 'vector_size': CNN_Encoder_ResNet50.vector_size,
                              'vectorize_func': vectorize_face_ResNet50, 'metric': 'angular'},
                             #{'preset_name': 'person_InceptionResNetV2',
                             # 'vector_size': CNN_Encoder_InceptionResNetV2.vector_size,
                             # 'vectorize_func': vectorize_face_InceptionResNetV2, 'metric': 'angular'},
                             {'preset_name': 'person_InceptionV3', 'vector_size': CNN_Encoder_InceptionV3.vector_size,
                              'vectorize_func': vectorize_face_InceptionV3, 'metric': 'angular'},
                             {'preset_name': 'person_ModileNetv3_Small',
                              'vector_size': CNN_encoder_ModileNetv3_Small.vector_size,
                              'vectorize_func': vectorize_face_ModileNetv3_Small, 'metric': 'angular'},
                             {'preset_name': 'person_CLIP_DML', 'vector_size': CNN_Encoder_CLIP_DML.vector_size,
                              'vectorize_func': vectorize_face_CLIP_DML, 'metric': 'angular'},
                             {'preset_name': 'person_DINO', 'vector_size': CNN_Encoder_DINO.vector_size,
                              'vectorize_func': vectorize_face_DINO, 'metric': 'angular'},
                             {'preset_name': 'person_BLIP', 'vector_size': CNN_Encoder_BLIP.vector_size,
                              'vectorize_func': vectorize_face_BLIP, 'metric': 'angular'},
                             #{'preset_name': 'person_custom', 'vector_size': CNN_Encoder_custom.vector_size,
                             #'vectorize_func': vectorize_face_custom, 'metric': 'angular'}
                             ]
                  }

    @property
    def is_wrong(self):
        return self.relation.get_field_val('is_wrong')

    @is_wrong.setter
    def is_wrong(self, value):
        self.relation.set_field_val('is_wrong', value)


for pref_name, pref_data in DubRelation.prefs_dict.items():
    for pref in pref_data:
        VectorDB.register_pref(pref['preset_name'], pref['vector_size'], pref['vectorize_func'], pref['metric'])
VectorDB.register_pref('FileRecord_ResNet50', CNN_Encoder_ResNet50.vector_size, vectorize_FileRecord_ResNet50,
                       'angular')
VectorDB.register_pref('FileRecord_ModileNetv3_Small', CNN_encoder_ModileNetv3_Small.vector_size,
                       vectorize_FileRecord_ModileNetv3_Small, 'angular')
VectorDB.register_pref('FileRecord_InceptionResNetV2', CNN_Encoder_InceptionResNetV2.vector_size,
                       vectorize_FileRecord_InceptionResNetV2, 'angular')
VectorDB.register_pref('FileRecord_InceptionV3', CNN_Encoder_InceptionV3.vector_size,
                       vectorize_FileRecord_InceptionV3, 'angular')
VectorDB.register_pref('FileRecord_CLIP_DML', CNN_Encoder_CLIP_DML.vector_size,
                       vectorize_FileRecord_CLIP_DML, 'angular')
VectorDB.register_pref('FileRecord_DINO', CNN_Encoder_DINO.vector_size,
                       vectorize_FileRecord_DINO, 'angular')
VectorDB.register_pref('FileRecord_BLIP', CNN_Encoder_BLIP.vector_size,
                       vectorize_FileRecord_BLIP, 'angular')
VectorDB.register_pref('FileRecord_custom', CNN_Encoder_custom.vector_size,
                       vectorize_FileRecord_custom, 'angular')
VectorDB.register_pref('FileRecord_mv2_custom', CNN_Encoder_mv2_custom.vector_size,
                       vectorize_FileRecord_mv2_custom, 'angular')
VectorDB.register_pref('FileRecord_fuse_MCBD', 768, vectorize_fileRecord_fuse_MCBD, 'angular')


def find_dubs_2(paths: list[str], related, threshold=0.95, format=CNN_encoder_ModileNetv3_Small.format,
                pats_dubs_search=True, related_search=True):
    format_to_preset = {CNN_Encoder_ResNet50.format: 'FileRecord_ResNet50',
                        CNN_encoder_ModileNetv3_Small.format: 'FileRecord_ModileNetv3_Small',
                        CNN_Encoder_InceptionResNetV2.format: 'FileRecord_InceptionResNetV2',
                        CNN_Encoder_InceptionV3.format: 'FileRecord_InceptionV3',
                        CNN_Encoder_CLIP_DML.format: 'FileRecord_CLIP_DML',
                        CNN_Encoder_DINO.format: 'FileRecord_DINO',
                        CNN_Encoder_BLIP.format: 'FileRecord_BLIP',
                        CNN_Encoder_custom.format: 'FileRecord_custom',
                        CNN_Encoder_mv2_custom.format: 'FileRecord_mv2_custom',
                        'FileRecord_fuse_MCBD': 'FileRecord_fuse_MCBD'}
    pref = VectorDB.get_pref(format_to_preset[format])
    all_files_pats = []
    for path in paths:
        list_records = get_file_record_by_folder(path, recurse=True)
        all_files_pats.extend(list_records)
    search_scope = SearchScopeList(pref, all_files_pats)
    dubs = []
    if pats_dubs_search:
        dubs = search_scope.find_dubs(5, threshold)
    if related_search:
        print("find related")
        for path in related:
            list_records = get_file_record_by_folder(path, recurse=True)
            for record in tqdm(list_records):
                res: ResultGroup = search_scope.search(record, 5, threshold)
                if len(res.results) > 0:
                    dubs.append(res)
    return dubs


def create_graf_dubs(paths: list[str], related: list[str], th=0.95, encoder=CNN_encoder_ModileNetv3_Small.format,
                     pats_dubs_search=True, related_search=True):
    emb_type = encoder
    dubslist = find_dubs_2(paths, related, th, emb_type, pats_dubs_search, related_search)
    bach_b = BatchBuilder(dubslist, 128)
    def exist_check(dub):
        source_file: FileRecord = dub.data_item
        ret_list = []
        if source_file is None:
            return ret_list
        for res in dub.results:
            res: ResultItem
            target_file:FileRecord = res.data_item
            if source_file is None or target_file is None or source_file == target_file or source_file.id == target_file.id or source_file.full_path == target_file.full_path:
                continue
            query = {"$or": [{'from_id': source_file._id, 'to_id': target_file._id, 'type': "similar_search"},
                             {'from_id': target_file._id, 'to_id': source_file._id, 'type': "similar_search"}]}
            exist = RelationRecord.find_one(query)
            if exist is not None:
                continue
            record = InsertOne({'from_id': source_file._id, 'to_id': target_file._id, 'type': "similar_search",
                                'sub_type': "none", 'emb_type': emb_type, 'distance': res.distance})
            ret_list.append(record)
        return ret_list

    def write_list_filtrate(write_list):
        exist_list = {}
        ret_list = []
        for record in write_list:
            source_file = str(record._doc['from_id'])
            target_file = str(record._doc['to_id'])
            if (source_file + target_file not in exist_list
                    and target_file + source_file not in exist_list):
                exist_list[source_file + target_file] = True
                ret_list.append(record)
        return ret_list

    for dub_list in tqdm(bach_b.bach.values()):
        write_list = []
        futures = []
        for dub in dub_list:
            futures.append(exist_check(dub))
        #with ThreadPoolExecutor(workers) as ex:
        #   futures = ex.map(exist_check, dub_list)

        for future in futures:
            if future is not None:
                write_list.extend(future)

        if len(write_list) > 0:
            write_list = write_list_filtrate(write_list)
            RelationRecord.collection().bulk_write(write_list)


def project_object_to_image_graps():
    query = {'type': 'similar_obj_search', 'sub_type': {'$ne': 'wrong'}}
    relations = RelationRecord.find(query)
    insert_list = []
    exist_from_to = {}
    for relation in tqdm(relations):
        sub_type = relation.get_field_val('sub_type')
        if sub_type == 'none':
            continue
        detecion_in = Detection(relation.from_id)
        detecion_out = Detection(relation.to_id)
        file_in = FileRecord(detecion_in.parent_image_id)
        file_out = FileRecord(detecion_out.parent_image_id)
        if str(file_in._id) + str(file_out._id) in exist_from_to:
            continue
        exist_from_to[str(file_in._id) + str(file_out._id)] = True
        if file_in._id == file_out._id:
            continue
        new_rel = InsertOne({'from_id': file_in._id, 'to_id': file_out._id, 'type': 'similar_search',
                             'sub_type': sub_type, 'emb_type': relation.get_field_val('emb_type'),
                             'distance': relation.get_field_val('distance')})

        exist = RelationRecord.find_one({'from_id': file_in._id, 'to_id': file_out._id, 'type': 'similar_search'})
        if exist is not None and exist.get_field_val('sub_type') != sub_type:
            RelationRecord.collection().delete_one({'_id': exist._id})
            insert_list.append(new_rel)
        if exist is None:
            insert_list.append(new_rel)
    if len(insert_list) > 0:
        print(len(insert_list))
        RelationRecord.collection().bulk_write(insert_list)


def del_image_search_refs(max_distance=0.95):
    query = {'type': 'similar_search', 'distance': {'$gt': max_distance}, 'sub_type': 'none'}
    RelationRecord.collection().delete_many(query)


def create_face_graph():
    for obj_class, pref_data in DubRelation.prefs_dict.items():
        for pref_setings in pref_data:
            pref = VectorDB.get_pref(pref_setings['preset_name'])
            obj_list = Detection.find({'object_class': obj_class,
                                       'is_wrong': {'$ne': True}})
            print(f'pref_setings: {pref_setings}')
            search_scope = SearchScopeList(pref, obj_list)
            dubs = search_scope.find_dubs(5, 0.4)
            exist = {}
            bach_write = []
            for obj_dub_group in tqdm(dubs):
                obj_dub_group: ResultGroup
                for obj_dub in obj_dub_group.results:
                    obj_dub: ResultItem
                    source: Detection = obj_dub_group.data_item
                    target: Detection = obj_dub.data_item

                    if source._id == target._id:
                        continue
                    if str(source._id) + str(target._id) in exist or str(target._id) + str(source._id) in exist:
                        continue
                    exist[str(source._id) + str(target._id)] = True
                    query = {"$or": [{'from_id': source._id, 'to_id': target._id, 'type': "similar_obj_search"},
                                     {'from_id': target._id, 'to_id': source._id, 'type': "similar_obj_search"}]}
                    resalt = RelationRecord.find_one(query)
                    if resalt is not None:
                        continue
                    record = InsertOne({'from_id': source._id, 'to_id': target._id, 'type': "similar_obj_search",
                                        'sub_type': "none", 'emb_type': pref.name, 'distance': obj_dub.distance,
                                        "object_class": obj_class})
                    bach_write.append(record)

            if len(bach_write) > 0:
                RelationRecord.collection().bulk_write(bach_write)


if __name__ == '__main__':
    Allocator.init_services()
    create_face_graph()
