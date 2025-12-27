import os

import cv2

from SLM import Allocator
from SLM.appGlue.DesignPaterns import MessageSystem
from SLM.appGlue.helpers import ImageHelper
from SLM.files_db.object_recognition.object_recognition import DetectionObjectClass, Detection, Recognized_object
from SLM.files_db.components.fs_tag import TagRecord


from SLM.indexerpyiplain.idexpyiplain import ItemIndexer, ItemFieldIndexer
from SLM.vision.objectdetectors import load_object_detectors
from SLM.vision.objectdetectors.object_detect import ObjectDetectorProvider

load_object_detectors()

class FaceDetector(ItemFieldIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MessageSystem.MessageSystem.Subscribe("ConfigChanged", self, self.on_config_change)
        self.fieldName = "face_detection"
        self.path =r"D:\data\ImageDataManager\Image_detect" #todo: move to tool class

    def on_config_change(self, section,attribute,value):
        if section == "Indexer" and attribute == "detectFaces":
            self.enabled = value

    def index(self, parent_indexer: ItemIndexer, item, need_index):
        from SLM.files_db.components.File_record_wraper import FileRecord

        record = FileRecord(item["_id"])
        #todo: refactor repository
        backends = Allocator.res.get_by_type_one(ObjectDetectorProvider)
        result_list = []
        for backend in backends:
            b_name = backend.format
            ind_backends = item.get("backend_indexed", [])
            if b_name in ind_backends:
                continue

            results = backend.detect_by_path(record.full_path)
            if len(results) != 0:
                for result in results:
                    width = result["region"][2]
                    height = result["region"][3]
                    if width < 20 or height < 20:
                        continue
                    result["backend"] = b_name
                    result_list.append(result)
            ind_backends.append(b_name)
            item["backend_indexed"] = ind_backends
            parent_indexer.shared_data["item_indexed"] = True

        # filtrate result list by comaring result regions
        # and select the best result by score

        boxes = cv2.dnn.NMSBoxes([result["region"] for result in result_list],
                                 [result["score"] for result in result_list],
                                 0.5, 0.4)
        if len(boxes) != 0:
            filtered_results = [result_list[i] for i in boxes]

            for result in filtered_results:

                backend_name = result["backend"]
                label = result["label"]
                score = result["score"]
                region = result["region"]
                region_format = result["region_format"]
                pil_image = result["image"]
                save_name = ImageHelper.content_md5(pil_image) + ".jpg"
                shard_folder = save_name[:2]
                save_path = os.path.join(self.path, shard_folder, save_name)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_image.save(save_path, "JPEG", quality=100)
                result["image_path"] = save_path
                image_path = result["image_path"]
                # create detection object
                obj_class = DetectionObjectClass.get_or_create(name=label)
                detection = Detection.new_record(object_class=obj_class.name, backend=backend_name, score=score,
                                                 rect_region=region, region_format=region_format,
                                                 obj_image_path=image_path)
                detection.parent_image_id = item["_id"]
                uncnown_recogn = Recognized_object.get_or_create(name="unknown")
                detection.set_recognized_object(uncnown_recogn)
                #detection.set_ref_to(item["_id"])

                # set tags for object
            tag = TagRecord.get_or_create(full_name="object_detect/face")
            tag.add_to_file_rec(record)


