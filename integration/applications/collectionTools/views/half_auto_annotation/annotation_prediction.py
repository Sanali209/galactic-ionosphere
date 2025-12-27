
import os.path
from typing import List




from SLM.appGlue.core import Service, Allocator
from SLM.files_db.annotation_tool.annotation import SLMAnnotationClient

from SLM.files_data_cache.imageToLabel import ImageToTextCache
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord

class Annotator:
    def __init__(self, name):
        self.name = name
        self.annotation_client = SLMAnnotationClient()

    def is_satisfied_by(self, candidate_label, item):
        if item.full_path is None:
            return False
        if item.full_path == "":
            return False
        if not os.path.exists(item.full_path):
            return False
        return True

    def is_compatible(self, job_name):
        return True




class AnnotationPredictionManager(Service):
    annotators = [Annotator("all_true")]

    def get_compatible_annotators(self, job_name) -> List[str]:
        return [x.name for x in self.annotators if x.is_compatible(job_name)]

    def get_annotator_by_name(self, name):
        for annotator_ in self.annotators:
            if annotator_.name == name:
                return annotator_
        return None

    def register(self):
        def decorator(cls):
            self.annotators.append(cls())
            return cls

        return decorator


Allocator.res.register( AnnotationPredictionManager())




@AnnotationPredictionManager.instance().register()
class metadata_im_genre_annotator(Annotator):
    def __init__(self):
        super().__init__("im genre metadata")

    def is_satisfied_by(self, candidate_label, item):
        # todo check to campacbiliti with fs_db
        if candidate_label == "3d renderer":
            for tag in item.tags:
                if tag.name == "manual|imgenre|3d renderer":
                    return True
                if tag.name == "manual|imgenre|sk bin filter|3d renderer":
                    return True
        if candidate_label == "photo":
            for tag in item.tags:
                if tag.name == "manual|imgenre|photho":
                    return True
                if tag.name == "manual|imgenre|photo":
                    return True
                if tag.name == "manual|imgenre|sk bin filter|photho":
                    return True
        if candidate_label == "drawing":
            for tag in item.tags:
                if tag.name == "manual|imgenre|drawn":
                    return True
                if tag.name == "manual|imgenre|anime":
                    return True

        return False

    def is_compatible(self, job_name):
        return job_name == "image genres"



@AnnotationPredictionManager.instance().register()
class Nsfw_boru_tags_annotator(Annotator):
    def __init__(self):
        super().__init__("NSFW Filter import boru data")

    def is_satisfied_by(self, candidate_label, item):
        tags = TagRecord.get_tags_of_file(item)
        for tag in tags:
            tag:TagRecord
            ind = tag.fullName.find("metadata/rating/"+candidate_label)
            if ind != -1:
                return True
        return False

    def is_compatible(self, job_name):
        return job_name == "NSFWFilter"

@AnnotationPredictionManager.instance().register()
class im_genre_HF_prediction_annotator(Annotator):
    def __init__(self):
        super().__init__("image genre HF")

    def is_satisfied_by(self, candidate_label, item):
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_genres_v001")
            if len(annotat_label) > 0:
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "image genres"


@AnnotationPredictionManager.instance().register()
class nsfw_filter_HF_prediction_annotator(Annotator):
    def __init__(self):
        super().__init__("nsfw filter HF")

    def is_satisfied_by(self, candidate_label, item):
        res = super().is_satisfied_by(candidate_label, item)
        if not res:
            return False
        try:
            path = item.full_path
            annotat_label = ImageToTextCache.instance().get_by_path(path, "multiclass_NSFW_HF")
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
class metadata_rating_annotator(Annotator):
    def __init__(self):
        super().__init__("rating metadata")

    def is_satisfied_by(self, candidate_label, item):
        res = super().is_satisfied_by(candidate_label, item)
        if not res:
            return False
        item: FileRecord
        if item.rating is None:
            return False
        if item.rating == 0:
            return False
        if candidate_label == "low":
            if 3 > item.rating > 0:
                return True
        if candidate_label == "normal":
            if item.rating == 3:
                return True
        if candidate_label == "high":
            if item.rating > 3:
                return True

        return False

    def is_compatible(self, job_name):
        return job_name == "rating"


@AnnotationPredictionManager.instance().register()
class metadata_comix_HF_prediction_annotator(Annotator):
    def __init__(self):
        super().__init__("comix HF")

    def is_satisfied_by(self, candidate_label, item):
        res = super().is_satisfied_by(candidate_label, item)
        if not res:
            return False
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_comix_bf")
            if len(annotat_label) > 0:
                if (annotat_label[0]['label'].startswith("3d")
                        and candidate_label == "3d renderer"):
                    return True
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "comiks binary"


@AnnotationPredictionManager().instance().register()
class metadata_image_type_HF_prediction_annotator(Annotator):
    def __init__(self):
        super().__init__("image type HF")

    def is_satisfied_by(self, candidate_label, item):
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_image_type_bf")
            if len(annotat_label) > 0:
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "image type"


@AnnotationPredictionManager.instance().register()
class rating_HF_prediction_annotator(Annotator):
    def __init__(self):
        super().__init__("rating HF")

    def is_satisfied_by(self, candidate_label, item):
        res = super().is_satisfied_by(candidate_label, item)
        if not res:
            return False
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_rating_HF")
            if len(annotat_label) > 0:
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "rating"


@AnnotationPredictionManager.instance().register()
class sketch_HF_annotator(Annotator):
    def __init__(self):
        super().__init__("sketch HF")

    def is_satisfied_by(self, candidate_label, item):
        try:
            annotat_label = ImageToTextCache.instance().get_by_path(item.full_path, "multiclass_sketch_bf")
            if len(annotat_label) > 0:
                annotat_label = annotat_label[0]['label']

            if annotat_label == candidate_label:
                return True
        except Exception as e:
            return False
        return False

    def is_compatible(self, job_name):
        return job_name == "sketch binary"
