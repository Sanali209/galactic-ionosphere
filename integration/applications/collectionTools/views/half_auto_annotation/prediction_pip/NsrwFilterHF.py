from SLM.files_data_cache.imageToLabel import ImageToTextCache
from SLM.files_db.components.fs_tag import TagRecord
from applications.collectionTools.views.half_auto_annotation.annotation_prediction import Annotator, \
    AnnotationPredictionManager
import json

# not in use

@AnnotationPredictionManager.instance().register()
class Nsfw_HF_annotator(Annotator):
    def __init__(self):
        super().__init__("NSFW Filter HF")

    def is_satisfied_by(self, candidate_label, item):
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_NSFW_HF")
            if len(annotat_label) > 0:
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "NSFWFilter"

@AnnotationPredictionManager.instance().register()
class Nsfw_boru_tags_annotator(Annotator):
    def __init__(self):
        super().__init__("NSFW Filter import boru data")

    def is_satisfied_by(self, candidate_label, item):
        tags = TagRecord.get_tags_of_file(item)
        for tag in tags:
            if tag.name == candidate_label:
                return True
        return False

    def is_compatible(self, job_name):
        return job_name == "NSFWFilter"



