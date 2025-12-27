from typing import Tuple

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from data_manager import data_manager
from constants import DEFAULT_MU


class TrueSkillAnnotationRecordTools:
    """TrueSkill tools using centralized cache"""

    @staticmethod
    def get_ts_values(ann_record: AnnotationRecord) -> Tuple[float, float]:
        """Get TrueSkill values for a record"""
        return data_manager.cache_manager.get_trueskill_values(str(ann_record.id))

    @staticmethod
    def clear_cache(ann_record: AnnotationRecord):
        """Clear cache for a record"""
        data_manager.cache_manager.clear_trueskill_cache(str(ann_record.id))

    @classmethod
    def set_all_sigma_to_default(cls, param: float):
        """Set all sigma values to default
        
        Note: This will skip anchors - their sigma values remain unchanged.
        """
        for rec in data_manager.manual_voted_list:
            # Skip anchors - they should not be modified
            if rec.get_field_val("ankor", False):
                continue

            # Update through data manager to maintain consistency
            current_mu = rec.get_field_val("avg_rating", DEFAULT_MU)
            if current_mu is None:
                current_mu = DEFAULT_MU
            data_manager.update_record_rating(rec, float(current_mu), param)
            cls.clear_cache(rec)

    @classmethod
    def set_all_mu_to_default(cls, mu_param: float, sigma_param: float):
        """Set all mu and sigma values to default
        
        Note: This will skip anchors - their values remain unchanged.
        """
        for rec in data_manager.manual_voted_list:
            # Skip anchors - they should not be modified
            if rec.get_field_val("ankor", False):
                continue

            # Update through data manager to maintain consistency
            data_manager.update_record_rating(rec, mu_param, sigma_param)
            cls.clear_cache(rec)

    @staticmethod
    def merge_items(praimary_record, merdge_record):
        list_of_subrecords = merdge_record.list_get("equal_list")
        list_of_subrecords.append(merdge_record.id)
        praimary_record.list_extend("equal_list", list_of_subrecords, True)
        merdge_record.set_field_val("manual", False)
        merdge_record.set_field_val("merged", True)
