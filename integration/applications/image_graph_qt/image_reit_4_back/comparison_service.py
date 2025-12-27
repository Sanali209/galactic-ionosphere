import random
from typing import Tuple, Optional, List

import trueskill
from tqdm import tqdm

from SLM.appGlue.core import ContextMode, ModeManager
from SLM.files_db.annotation_tool.annotation import AnnotationRecord

from anotation_tool import TrueSkillAnnotationRecordTools
from constants import ACESSIBLE_QUALITY, MODEL_SIGMA, ANTI_STACK_COUNT, \
    DEFAULT_MU
from data_manager import data_manager
from loguru import logger


class BaseAnnotationMode(ContextMode):
    """Base class for annotation modes"""

    def get_next_pair(self) -> Tuple[Optional[AnnotationRecord], Optional[AnnotationRecord]]:
        return None, None


class Get_random_pair(BaseAnnotationMode):
    """Random pair selection mode"""
    vote_list: List[AnnotationRecord] = []
    activated = False

    def activate(self):
        Get_random_pair.vote_list = data_manager.manual_voted_list.copy()

    def get_next_pair(self) -> Tuple[Optional[AnnotationRecord], Optional[AnnotationRecord]]:
        first = self.get_pair_with_lowest_count()
        second = self.get_good_quality_pair(first)

        if first is None or second is None:
            return None, None
        if first.file is None or second.file is None:
            return None, None

        return first, second

    def get_pair_with_lowest_count(self) -> Optional[AnnotationRecord]:
        if not Get_random_pair.activated or len(Get_random_pair.vote_list) == 0:
            Get_random_pair.activate(self)
            Get_random_pair.activated = True
        if len(Get_random_pair.vote_list) == 0:
            return None
        return_item = random.choice(Get_random_pair.vote_list)
        Get_random_pair.vote_list.remove(return_item)
        return return_item

    def get_good_quality_pair(self, first: AnnotationRecord) -> Optional[AnnotationRecord]:
        choices = data_manager.manual_voted_list.copy()
        if len(choices) == 0:
            return None

        first_mu, first_sigma = TrueSkillAnnotationRecordTools.get_ts_values(first)
        first_rating = trueskill.Rating(mu=first_mu, sigma=first_sigma)

        st_counter = 100
        while st_counter > 0:
            item = random.choice(choices)
            if first.id == item.id:
                continue

            i_mu, i_sigma = TrueSkillAnnotationRecordTools.get_ts_values(item)
            quality = trueskill.quality_1vs1(first_rating, trueskill.Rating(mu=i_mu, sigma=i_sigma))

            if quality > ACESSIBLE_QUALITY:
                return item
            st_counter -= 1

        # Return any valid item if quality threshold not met
        return random.choice([r for r in choices if r.id != first.id])


class pair_sel_bigest_sigma(BaseAnnotationMode):
    """Biggest sigma pair selection mode"""
    last_1st_record: Optional[AnnotationRecord] = None
    staced_count = 0
    work_list: List[AnnotationRecord] = []
    choices: List[AnnotationRecord] = []

    def activate(self, *args, **kwargs):
        if len(data_manager.manual_voted_list) == 0:
            return

        with tqdm(total=len(data_manager.manual_voted_list), desc="Selecting image pair") as progress:
            def get_key(record: AnnotationRecord) -> float:
                progress.update(1)
                sigma = record.get_field_val("trueskill_sigma", MODEL_SIGMA)
                return sigma

            data_manager.manual_voted_list.sort(key=get_key, reverse=True)
            pair_sel_bigest_sigma.work_list = data_manager.manual_voted_list.copy()
            pair_sel_bigest_sigma.choices = data_manager.manual_voted_list.copy()

    def get_next_pair(self) -> Tuple[Optional[AnnotationRecord], Optional[AnnotationRecord]]:
        if len(pair_sel_bigest_sigma.work_list) == 0:
            return None, None

        # Sort by sigma — descending uncertainty
        pair_sel_bigest_sigma.work_list.sort(
            key=lambda r: r.get_field_val("trueskill_sigma", MODEL_SIGMA), reverse=True
        )

        # 5% chance to pick randomly, otherwise pick with highest sigma
        if random.random() < 0.05:
            img_rec1 = random.choice(pair_sel_bigest_sigma.work_list)
        else:
            img_rec1 = pair_sel_bigest_sigma.work_list[0]

        # Anti-stacking logic
        if pair_sel_bigest_sigma.last_1st_record == img_rec1:
            pair_sel_bigest_sigma.staced_count += 1
            if pair_sel_bigest_sigma.staced_count > ANTI_STACK_COUNT:
                pair_sel_bigest_sigma.staced_count = 0
                pair_sel_bigest_sigma.work_list.remove(img_rec1)
                return self.get_next_pair()
        else:
            pair_sel_bigest_sigma.last_1st_record = img_rec1
            pair_sel_bigest_sigma.staced_count = 0

        image2 = self.get_good_quality_pair(img_rec1)
        return img_rec1, image2

    def get_good_quality_pair(self, first: AnnotationRecord) -> Optional[AnnotationRecord]:
        first_mu, first_sigma = TrueSkillAnnotationRecordTools.get_ts_values(first)
        first_rating = trueskill.Rating(mu=first_mu, sigma=first_sigma)

        count = 100
        while count > 0:
            item = random.choice(pair_sel_bigest_sigma.choices)
            if first.id == item.id:
                continue

            i_mu, i_sigma = TrueSkillAnnotationRecordTools.get_ts_values(item)
            quality = trueskill.quality_1vs1(first_rating, trueskill.Rating(mu=i_mu, sigma=i_sigma))

            if quality > ACESSIBLE_QUALITY:
                return item
            count -= 1

        # Return any valid item if quality threshold not met
        valid_choices = [r for r in pair_sel_bigest_sigma.choices if r.id != first.id]
        return random.choice(valid_choices) if valid_choices else None




class pair_sel_by_hardcoded_ankors(BaseAnnotationMode):
    """Pair selection with hardcoded anchors mode"""
    last_1st_record: Optional[AnnotationRecord] = None
    staced_count = 0
    ankors: List[AnnotationRecord] = []
    not_ankors_procesing: List[AnnotationRecord] = []
    not_ankors: List[AnnotationRecord] = []

    def activate(self, *args, **kwargs):
        self.ankors.clear()
        self.not_ankors.clear()
        if len(data_manager.manual_voted_list) == 0:
            return

        all_records = data_manager.manual_voted_list.copy()

        for record in all_records:
            if record.get_field_val("ankor", False, hashed=False):
                self.ankors.append(record)
            else:
                self.not_ankors.append(record)

        if not self.ankors:
            logger.warning("No hardcoded anchors found. Please mark some records as anchors.")
            return

        self.ankors.sort(key=lambda r: r.get_field_val("avg_rating", DEFAULT_MU))
        self.not_ankors_procesing = self.not_ankors.copy()

    def get_next_pair(self) -> Tuple[Optional[AnnotationRecord], Optional[AnnotationRecord]]:
        if not self.not_ankors or not self.ankors:
            return None, None

        if len(self.not_ankors_procesing) == 0:
            self.ankors.sort(key=lambda r: r.get_field_val("avg_rating", DEFAULT_MU))
            self.not_ankors_procesing.extend(self.not_ankors)

        img_rec1 = self.not_ankors_procesing.pop(0)

        img_rec2 = self.get_good_ankor(img_rec1)

        if not img_rec2:
            return None, None

        return img_rec1, img_rec2

    def get_good_ankor(self, first: AnnotationRecord) -> Optional[AnnotationRecord]:
        first_mu, _ = TrueSkillAnnotationRecordTools.get_ts_values(first)
        nearest_delta = float('inf')
        nearest_anchor = None

        for anchor in self.ankors:
            anchor_mu, _ = TrueSkillAnnotationRecordTools.get_ts_values(anchor)
            delta = abs(first_mu - anchor_mu)
            if delta < nearest_delta and first.id != anchor.id:
                nearest_delta = delta
                nearest_anchor = anchor

        return nearest_anchor


class pair_sel_hardcoded_anchors_high_sigma(BaseAnnotationMode):
    """Pair selection with hardcoded anchors, only comparing images with sigma > 1"""
    last_1st_record: Optional[AnnotationRecord] = None
    staced_count = 0
    ankors: List[AnnotationRecord] = []
    high_sigma_records: List[AnnotationRecord] = []
    high_sigma_processing: List[AnnotationRecord] = []

    def activate(self, *args, **kwargs):
        self.ankors.clear()
        self.high_sigma_records.clear()
        if len(data_manager.manual_voted_list) == 0:
            return

        all_records = data_manager.manual_voted_list.copy()

        # Separate anchors from regular records
        for record in all_records:
            if record.get_field_val("ankor", False, hashed=False):
                self.ankors.append(record)
            else:
                # Only include records with sigma > 1
                sigma = record.get_field_val("trueskill_sigma", MODEL_SIGMA)
                if sigma > 1:
                    self.high_sigma_records.append(record)

        if not self.ankors:
            logger.warning("No hardcoded anchors found. Please mark some records as anchors.")
            return

        if not self.high_sigma_records:
            logger.warning("No records found with sigma > 1")
            return

        # Sort anchors by rating for consistent pairing
        self.ankors.sort(key=lambda r: r.get_field_val("avg_rating", DEFAULT_MU))
        self.high_sigma_processing = self.high_sigma_records.copy()

        logger.info(
            f"Activated high sigma anchors mode: {len(self.ankors)} anchors, {len(self.high_sigma_records)} high-sigma records")

    def get_next_pair(self) -> Tuple[Optional[AnnotationRecord], Optional[AnnotationRecord]]:
        if not self.high_sigma_records or not self.ankors:
            return None, None

        if len(self.high_sigma_processing) == 0:
            # Reset processing list when exhausted
            self.high_sigma_processing = self.high_sigma_records.copy()

        img_rec1 = self.high_sigma_processing.pop(0)

        img_rec2 = self.get_good_ankor(img_rec1)

        if not img_rec2:
            logger.warning(f"No suitable anchor found for record with sigma > 1")
            return None, None

        return img_rec1, img_rec2

    def get_good_ankor(self, first: AnnotationRecord) -> Optional[AnnotationRecord]:
        """Find the nearest anchor by rating for the given record"""
        first_mu, _ = TrueSkillAnnotationRecordTools.get_ts_values(first)
        nearest_delta = float('inf')
        nearest_anchor = None

        for anchor in self.ankors:
            anchor_mu, _ = TrueSkillAnnotationRecordTools.get_ts_values(anchor)
            delta = abs(first_mu - anchor_mu)
            if delta < nearest_delta and first.id != anchor.id:
                nearest_delta = delta
                nearest_anchor = anchor

        return nearest_anchor


class ComparisonService:
    """Service for managing comparison operations"""
    cursor: int = 0
    select_strategy = None
    select_strategy_provider = None
    select_pair_mode_manager = ModeManager()
    select_pair_mode_manager.register_mode("Random", Get_random_pair)
    select_pair_mode_manager.register_mode("get bigest sigma", pair_sel_bigest_sigma)
    select_pair_mode_manager.register_mode("Pair with Hardcoded Anchors", pair_sel_by_hardcoded_ankors)
    select_pair_mode_manager.register_mode("Hardcoded Anchors (Sigma > 1)", pair_sel_hardcoded_anchors_high_sigma)
