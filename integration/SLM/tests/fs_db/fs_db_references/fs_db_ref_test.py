import os

import loguru
from annoy import AnnoyIndex
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.appGlue.iotools.pathtools import get_files, get_sub_dirs
from SLM.files_data_cache.tensor import Embeddings_cache
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.relations.dubsearch import create_graf_dubs, del_image_search_refs, create_face_graph, \
    project_object_to_image_graps
import unittest

from SLM.files_db.components.relations.relation import RelationRecord
from SLM.files_db.object_recognition.object_recognition import Detection
from SLM.vision.imagetotensor.CNN_Encoding import ImageToCNNTensor
from SLM.vision.imagetotensor.backends.BLIP import CNN_Encoder_BLIP
from SLM.vision.imagetotensor.backends.DINO import CNN_Encoder_DINO
from SLM.vision.imagetotensor.backends.clip_vit_dirml import CNN_Encoder_CLIP_DML
from SLM.vision.imagetotensor.backends.inceptionV3 import CNN_Encoder_InceptionV3
from SLM.vision.imagetotensor.backends.inception_resnet_v2 import CNN_Encoder_InceptionResNetV2
from SLM.vision.imagetotensor.backends.resnet import CNN_Encoder_ResNet50
from SLM.vision.imagetotensor.custom.custom_emb import CNN_Encoder_custom

config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_modules()


class TestDBRefs(unittest.TestCase):

    def test_find_dubs(self):

        pats_list = [[r"E:\rawimagedb\repository\nsfv repo\drawn\presort\_by races"
                      ]]

        get_from_path = []  #r'E:\rawimagedb\repository\_art books',
        #r'E:\rawimagedb\repository\nsfv repo\furi\furi sites rip',
        #r"E:\rawimagedb\repository\nsfv repo\furi\Furry games",
        #r'E:\rawimagedb\repository\nsfv repo\furi\furi autors',
        #r'E:\rawimagedb\repository\nsfv repo\furi\Yiff Artworks',
        #r'E:\rawimagedb\repository\nsfv repo\drawn\_site rip',
        #r'E:\rawimagedb\r  epository\nsfv repo\drawn\drawn xxx autors',
        #r'E:\rawimagedb\repository\nsfv repo\drawn\presort\_cartoons\by name',
        #r'E:\rawimagedb\repository\nsfv repo\drawn\presort\_films',
        #r'E:\rawimagedb\repository\nsfv repo\3d\_siterips']
        for fp in get_from_path:
            # get list of folder of get_from pats and write to path_list
            subpats = get_sub_dirs(fp)
            for path in subpats:
                pats_list.append([path])
        related = []  #r"E:\rawimagedb\repository\nsfv repo\drawn\presort"]
        pbar = tqdm(pats_list)
        for path0 in pbar:
            pbar.set_description(f"Processing {path0}")
            print("castom")
            create_graf_dubs(path0, related, 0.3, encoder=CNN_Encoder_custom.format, pats_dubs_search=True,
                             related_search=False)
            print("castom")
            #create_graf_dubs(path0, related, 0.4,encoder="FileRecord_fuse_MCBD", pats_dubs_search=True,
            #related_search=False)
            print("mobilenet")
            #create_graf_dubs(path0, related, 0.4, pats_dubs_search=True, related_search=False)
            print("ResNet50")
            #create_graf_dubs(path0, related, 0.34, encoder=CNN_Encoder_ResNet50.format, pats_dubs_search=True,
            #related_search=False)
            print("clip")
            create_graf_dubs(path0, related, 0.4, encoder=CNN_Encoder_CLIP_DML.format, pats_dubs_search=True,
                             related_search=False)
            print("InceptionV3")
            #create_graf_dubs(path0, related, 0.4, encoder=CNN_Encoder_InceptionV3.format, pats_dubs_search=True,
            #related_search=False)
            print("InceptionResNetV2")
            #create_graf_dubs(path0, related, 0.25, encoder=CNN_Encoder_InceptionResNetV2.format, pats_dubs_search=True,
            # related_search=False)

            print("Dino")
            #create_graf_dubs(path0, related, 0.1, encoder=CNN_Encoder_DINO.format, pats_dubs_search=True,
            #related_search=False)
            print("BLIP")
            create_graf_dubs(path0, related, 0.4, encoder=CNN_Encoder_BLIP.format, pats_dubs_search=True,
                             related_search=False)

    def test_del_custom_rels(self):
        del_path = r"E:\rawimagedb\repository"
        relations = RelationRecord.find(
            {'type': "similar_search", "emb_type": "custom", "sub_type": "none", "distance": {"$gt": 0.000001}})
        for relation in tqdm(relations):
            relation: RelationRecord
            print(f"del")
            relation.delete_rec()

    def test_encoders_diagrams_for_image_refs(self):
        mesurment_template = {"angular min": 0,
                              "angular max": 0,
                              "euclidean min": 0,
                              "euclidean max": 0,
                              "manhattan min": 0,
                              "manhattan max": 0,
                              "hamming min": 0,
                              "hamming max": 0,
                              "dot min": 0,
                              "dot max": 0,
                              }
        diagrams_data = {"CLIP": {"count": 0,
                                  "wrong_count": 0,
                                  "not_wrong_count": 0,
                                  "wrong": mesurment_template.copy(),
                                  "similar": mesurment_template.copy(),
                                  "near_dub": mesurment_template.copy(),
                                  "similar_style": mesurment_template.copy(),
                                  "some_person": mesurment_template.copy(),
                                  "some_image_set": mesurment_template.copy(),
                                  "other": mesurment_template.copy(),
                                  "none": mesurment_template.copy(),
                                  },
                         }
        relations = RelationRecord.find({'type': "similar_search"})
        for relation in tqdm(relations):
            type = relation.get_field_val("emb_type")
            if type == "manual":
                continue
            cur_diagram_data = diagrams_data.get(type)
            if cur_diagram_data is None:
                continue

    def test_del_refs(self):
        del_image_search_refs(0.3)

    def test_measure_additional_distances_for_image_refs(self):
        relations = RelationRecord.find({'type': "similar_search"})
        tensor_cache = Embeddings_cache(["ModileNetV3Small", CNN_Encoder_CLIP_DML.format,
                                         CNN_Encoder_ResNet50.format, CNN_Encoder_InceptionResNetV2.format])
        for relation in tqdm(relations):
            from_rec = FileRecord(relation.from_id)
            to_rec = FileRecord(relation.to_id)
            type = relation.get_field_val("emb_type")
            if type == "manual":
                continue
            exist = relation.get_field_val("euclidean")
            if exist is not None:
                continue
            if type == "CLIP":
                type = CNN_Encoder_CLIP_DML.format
            from_tensor = tensor_cache.get_by_path(from_rec.full_path, type)
            to_tensor = tensor_cache.get_by_path(to_rec.full_path, type)
            vector_size = ImageToCNNTensor.all_backends[type].vector_size
            indexer_e = AnnoyIndex(vector_size, "euclidean")
            indexer_m = AnnoyIndex(vector_size, "manhattan")
            indexer_h = AnnoyIndex(vector_size, "hamming")
            indexer_d = AnnoyIndex(vector_size, "dot")
            indexer_e.add_item(0, from_tensor)
            indexer_m.add_item(0, from_tensor)
            indexer_h.add_item(0, from_tensor)
            indexer_d.add_item(0, from_tensor)
            indexer_e.add_item(1, to_tensor)
            indexer_m.add_item(1, to_tensor)
            indexer_h.add_item(1, to_tensor)
            indexer_d.add_item(1, to_tensor)

            indexer_e.build(vector_size)
            indexer_m.build(vector_size)
            indexer_h.build(vector_size)
            indexer_d.build(vector_size)
            dist_e = indexer_e.get_distance(0, 1)
            dist_m = indexer_m.get_distance(0, 1)
            dist_h = indexer_h.get_distance(0, 1)
            dist_d = indexer_d.get_distance(0, 1)
            relation.set_field_val("euclidean", dist_e)
            relation.set_field_val("manhattan", dist_m)
            relation.set_field_val("hamming", dist_h)
            relation.set_field_val("dot", dist_d)

    def test_clip_import(self):
        path = r"G:\Мой диск\CLIP_DML"
        tensor_cache = Embeddings_cache([CNN_Encoder_CLIP_DML.format])
        tensor_cache.import_from_other_cache(path, CNN_Encoder_CLIP_DML.format)

    def test_face_graph(self):
        create_face_graph()

    def test_project_object_to_persons(self):
        project_object_to_image_graps()

    def test_optimize_image_dubSearchRelations(self):

        relations = RelationRecord.find({'type': "similar_search"})
        for relation in tqdm(relations):
            relation: RelationRecord
            from_rec: FileRecord = FileRecord.find_one({"_id": relation.from_id})
            to_rec: FileRecord = FileRecord.find_one({"_id": relation.to_id})
            if from_rec is None or to_rec is None:
                loguru.logger.info("From or To record is None, deleting relation")
                relation.delete_rec()
                continue
            if not os.path.exists(from_rec.full_path):
                loguru.logger.info(f"From record path does not exist: {from_rec.full_path}, deleting relation")
                relation.delete_rec()
                from_rec.delete_rec()

            if not os.path.exists(to_rec.full_path):
                loguru.logger.info(f"To record path does not exist: {to_rec.full_path}, deleting relation")
                relation.delete_rec()
                to_rec.delete_rec()

            if from_rec.full_path == to_rec.full_path:
                loguru.logger.info(f"From and To records are the same: {from_rec.full_path}, deleting relation")
                relation.delete_rec()
                if from_rec._id != to_rec._id:
                    to_rec.delete_rec()
                continue

    def test_optimize_obj_dubSearchRelations(self):

        relations = RelationRecord.find({'type': "similar_obj_search"})
        for relation in tqdm(relations):

            from_rec: Detection = Detection.find_one({"_id": relation.from_id})
            to_rec: Detection = Detection.find_one({"_id": relation.to_id})
            if from_rec is None or to_rec is None or from_rec.parent_file is None or to_rec.parent_file is None:
                relation.delete_rec()
                continue
            if from_rec.parent_file.full_path is None or to_rec.parent_file.full_path is None:
                relation.delete_rec()
                continue
            if not os.path.exists(from_rec.parent_file.full_path):
                relation.delete_rec()
                from_rec.delete_rec()

            if not os.path.exists(to_rec.parent_file.full_path):
                relation.delete_rec()
                to_rec.delete_rec()

            if from_rec.parent_file.full_path == to_rec.parent_file.full_path:
                relation.delete_rec()
                if from_rec.parent_file._id != to_rec.parent_file._id:
                    to_rec.delete_rec()
                continue
