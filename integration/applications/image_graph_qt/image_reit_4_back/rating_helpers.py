"""
Rating calculation and pair processing helpers
"""
from typing import Optional, Tuple, List
from loguru import logger
import trueskill
from tqdm import tqdm

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from anotation_tool import TrueSkillAnnotationRecordTools
from constants import DEFAULT_MU, MODEL_SIGMA


class RatingHelpers:
    """Static helper methods for rating calculations and pair processing"""

    @staticmethod
    def denormalize_rating(predicted_rating: float, transform_mean: float, transform_std_dev: float) -> float:
        """
        Convert normalized rating (1-10 scale) back to TrueSkill mu value
        
        Uses simple linear denormalization since the model output is already 0-1 and scaled to 1-10.
        The transform_mean represents the mu value at rating 1, and transform_std_dev is used as range.
        
        Args:
            predicted_rating: Rating value from model (1-10 scale)
            transform_mean: Mean mu value (corresponds to rating ~5.5)
            transform_std_dev: Standard deviation of mu values (used for range)
            
        Returns:
            Denormalized mu value
        """
        # Clamp rating to valid range
        predicted_rating = max(1.0, min(10.0, predicted_rating))
        
        # Simple linear mapping: rating 1-10 maps to mu range
        # Estimate mu range as mean ± 3*std_dev (covers ~99.7% of data)
        min_mu = transform_mean - 3.0 * transform_std_dev
        max_mu = transform_mean + 3.0 * transform_std_dev
        
        # Linear interpolation from rating to mu
        mu = min_mu + ((predicted_rating - 1.0) / 9.0) * (max_mu - min_mu)
        
        logger.debug(f"Denormalize: rating={predicted_rating:.2f} → mu={mu:.2f} "
                    f"(range: {min_mu:.2f} to {max_mu:.2f}, mean={transform_mean:.2f}, std={transform_std_dev:.2f})")
        
        return mu

    @staticmethod
    def parse_pair_key(key) -> Optional[Tuple[str, str]]:
        """Parse pair cache key to extract two record IDs

        Args:
            key: Cache key in format "id1_id2" or "complex_id_with_underscores_id2"

        Returns:
            Tuple of (first_id, second_id) or None if invalid
        """
        # Convert key to string if it isn't already
        key_str = str(key)

        parts = key_str.split('_')
        if len(parts) < 2:
            return None

        first_id = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
        second_id = parts[-1]
        return first_id, second_id

    @staticmethod
    def find_record_by_id(record_id: str) -> Optional[AnnotationRecord]:
        """Find annotation record by ID from a list

        Args:
            record_id: The ID to search for
            records_list: List of records to search in

        Returns:
            AnnotationRecord if found, None otherwise
        """
        return AnnotationRecord(record_id)

    @staticmethod
    def calculate_conservative_rating(record: AnnotationRecord) -> float:
        """Calculate conservative rating (mu - 3*sigma) for a record

        Args:
            record: The annotation record

        Returns:
            Conservative rating value
        """
        mu = record.get_field_val("avg_rating", DEFAULT_MU)
        sigma = record.get_field_val("trueskill_sigma", MODEL_SIGMA)
        return mu - 3 * sigma

    @staticmethod
    def calculate_trueskill_with_fallback(win_rating: trueskill.Rating,
                                          lose_rating: trueskill.Rating) -> Tuple[trueskill.Rating, trueskill.Rating]:
        """Calculate TrueSkill ratings with mpmath fallback

        Args:
            win_rating: Winner's current rating
            lose_rating: Loser's current rating

        Returns:
            Tuple of (new_win_rating, new_lose_rating)
        """
        try:
            return trueskill.rate_1vs1(win_rating, lose_rating)
        except Exception as e:
            logger.warning(f"TrueSkill calculation failed, using mpmath backend: {e}")
            ts = trueskill.TrueSkill(backend="mpmath")
            return ts.rate_1vs1(win_rating, lose_rating)

    @staticmethod
    def collect_valid_pairs_from_cache(cache_manager, records_list: List[AnnotationRecord],
                                       skip_anchors: bool = False) -> List[Tuple[AnnotationRecord, AnnotationRecord]]:
        """Collect valid pairs from cache, avoiding duplicates

        Args:
            cache_manager: The cache manager instance
            records_list: List of records to process
            skip_anchors: If True, skip pairs involving anchor records

        Returns:
            List of (winner, loser) tuples
        """
        pairs_cache = cache_manager.pairs_cache
        if not pairs_cache:
            return []

        processed_pairs = set()
        valid_pairs = []

        for key in tqdm(pairs_cache):
            winner_id = pairs_cache.get(key)
            if not winner_id:
                continue

            # Parse pair key
            pair_ids = RatingHelpers.parse_pair_key(key)
            if not pair_ids:
                continue

            first_id, second_id = pair_ids

            # Avoid duplicate processing
            normalized_key = tuple(sorted([first_id, second_id]))
            if normalized_key in processed_pairs:
                continue
            processed_pairs.add(normalized_key)

            # Find records
            winner_rec = RatingHelpers.find_record_by_id(winner_id)
            # todo can be optimized by use AnnotationRecord class
            loser_id = first_id if first_id != winner_id else second_id
            loser_rec = RatingHelpers.find_record_by_id(loser_id)

            if not winner_rec or not loser_rec:
                continue

            # Skip anchors if requested
            if skip_anchors:
                winner_is_anchor = winner_rec.get_field_val("is_ankor", False)
                loser_is_anchor = loser_rec.get_field_val("is_ankor", False)
                if winner_is_anchor or loser_is_anchor:
                    continue

            valid_pairs.append((winner_rec, loser_rec))

        return valid_pairs

    @staticmethod
    def process_pair_with_trueskill(winner_rec: AnnotationRecord, loser_rec: AnnotationRecord,
                                    data_manager) -> bool:
        """Process a single pair with TrueSkill rating update

        Args:
            winner_rec: Winner record
            loser_rec: Loser record
            data_manager: Data manager instance for updating ratings

        Returns:
            True if successful, False otherwise
            
        Note: If either record is an anchor, the pair will be processed but anchor values won't be modified.
        """
        try:
            # Check if either is an anchor
            winner_is_anchor = winner_rec.get_field_val("ankor", False)
            loser_is_anchor = loser_rec.get_field_val("ankor", False)
            
            # Get current ratings
            win_mu, win_sigma = TrueSkillAnnotationRecordTools.get_ts_values(winner_rec)
            lose_mu, lose_sigma = TrueSkillAnnotationRecordTools.get_ts_values(loser_rec)

            # Calculate new ratings
            win_rating = trueskill.Rating(mu=win_mu, sigma=win_sigma)
            lose_rating = trueskill.Rating(mu=lose_mu, sigma=lose_sigma)

            new_win, new_lose = RatingHelpers.calculate_trueskill_with_fallback(win_rating, lose_rating)

            if not winner_is_anchor:
                data_manager.update_record_rating(winner_rec, new_win.mu, new_win.sigma)
            if not loser_is_anchor:
                data_manager.update_record_rating(loser_rec, new_lose.mu, new_lose.sigma)

            return True
        except Exception as e:
            logger.error(f"Error processing pair {winner_rec.id} vs {loser_rec.id}: {e}")
            return False
