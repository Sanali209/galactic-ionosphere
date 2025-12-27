"""
Rating Service - Unified rating calculations following Single Source of Truth
Consolidates all rating logic from rating_helpers.py, rating_operations.py, and inline calculations.
"""

import random
from typing import List, Optional, Tuple, Dict
import numpy as np

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from loguru import logger
import trueskill

from services.configuration_service import ConfigurationService


class RatingService:
    """Unified rating calculations and TrueSkill operations"""

    def __init__(self):
        # Lazy load dependencies
        from services.service_container import service_container
        self._config_service = service_container.get_service(ConfigurationService)
        self._data_service = None
        self._cache_service = None

        logger.info("RatingService initialized")

    def _get_data_service(self):
        """Lazy load data service"""
        if self._data_service is None:
            from services.service_container import service_container
            from services.data_service import DataService
            self._data_service = service_container.get_service(DataService)
        return self._data_service

    def _get_cache_service(self):
        """Lazy load cache service"""
        if self._cache_service is None:
            from services.service_container import service_container
            from .cache_service import CacheService
            self._cache_service = service_container.get_service(CacheService)
        return self._cache_service

    def get_conservative_rating(self, record: AnnotationRecord) -> float:
        """Calculate conservative rating (μ - 3σ)"""
        config = self._config_service.get_config()
        mu = record.get_field_val("avg_rating", config.DEFAULT_MU)
        sigma = record.get_field_val("trueskill_sigma", config.MODEL_SIGMA)
        return mu - 3 * sigma

    def calculate_average_conservative_rating(self, records: List[AnnotationRecord]) -> float:
        """Calculate average conservative rating from a list of records"""
        if not records:
            config = self._config_service.get_config()
            return config.DEFAULT_MU

        conservative_ratings = [self.get_conservative_rating(rec) for rec in records]
        return sum(conservative_ratings) / len(conservative_ratings)

    def update_record_rating(self, record: AnnotationRecord, mu: float, sigma: float,
                           update_cache: bool = True):
        """Update a record's rating with TrueSkill values"""
        config = self._config_service.get_config()

        # Update database fields
        record.set_field_val("avg_rating", mu)
        record.set_field_val("trueskill_sigma", sigma)
        record.set_field_val("manual", True)

        if update_cache:
            cache_service = self._get_cache_service()
            record_id = str(record.id)
            cache_service.set_trueskill_values(record_id, mu, sigma)
            cache_service.set_item_rating(record_id, mu, sigma)

        logger.debug(f"Updated rating for record {record.id}: μ={mu:.2f}, σ={sigma:.2f}")

    def process_comparison_result(self, winner: AnnotationRecord, loser: AnnotationRecord,
                                update_cache: bool = True) -> bool:
        """Process a single comparison result using TrueSkill"""
        try:
            # Get current ratings
            win_mu, win_sigma = self._get_true_values(winner)
            lose_mu, lose_sigma = self._get_true_values(loser)

            # Create TrueSkill ratings
            win_rating = trueskill.Rating(mu=win_mu, sigma=win_sigma)
            lose_rating = trueskill.Rating(mu=lose_mu, sigma=lose_sigma)

            # Calculate new ratings
            new_win, new_lose = trueskill.rate_1vs1(win_rating, lose_rating)

            # Update both records
            self.update_record_rating(winner, new_win.mu, new_win.sigma, update_cache)
            self.update_record_rating(loser, new_lose.mu, new_lose.sigma, update_cache)

            # Cache the result if requested
            if update_cache:
                cache_service = self._get_cache_service()
                cache_service.set_pair_winner(str(winner.id), str(loser.id), str(winner.id))

            logger.debug(f"Processed comparison: {winner.id} beat {loser.id}")
            return True

        except Exception as e:
            logger.error(f"Error processing comparison result: {e}")
            return False

    def process_multi_way_ranking(self, rankings: List[List[AnnotationRecord]],
                                update_cache: bool = True) -> bool:
        """Process N-way ranking (e.g., 6-image ranking)"""
        try:
            if not rankings:
                logger.warning("No rankings provided for multi-way processing")
                return False

            # Create TrueSkill ratings for all records
            ratings = []
            records = []

            for ranking_list in rankings:
                for record in ranking_list:
                    if record not in records:  # Avoid duplicates
                        records.append(record)
                        mu, sigma = self._get_true_values(record)
                        ratings.append(trueskill.Rating(mu=mu, sigma=sigma))

            if len(ratings) < 2:
                logger.warning("Need at least 2 records for ranking")
                return False

            # Calculate new ratings using N-way ranking
            new_ratings = trueskill.rate([tuple(ratings)])  # All records ranked within single tier

            # Update all records with new ratings
            for i, record in enumerate(records):
                self.update_record_rating(record, new_ratings[0][i].mu, new_ratings[0][i].sigma, update_cache)

            logger.info(f"Processed {len(records)}-way ranking")
            return True

        except Exception as e:
            logger.error(f"Error processing multi-way ranking: {e}")
            return False

    def normalize_ratings_zscore(self, records: List[AnnotationRecord],
                               midpoint_record: Optional[AnnotationRecord] = None,
                               update_records: bool = True) -> Dict[str, float]:
        """
        Normalize ratings using Z-score method

        Args:
            records: Records to normalize
            midpoint_record: Optional record to use as midpoint
            update_records: Whether to update records or just calculate

        Returns:
            Dictionary with normalization statistics
        """
        try:
            if not records:
                return {'error': 'No records provided'}

            # Get current ratings
            values = [rec.get_field_val("avg_rating", 25.0) for rec in records]

            if len(values) < 2:
                logger.warning("Need at least 2 records for normalization")
                return {'error': 'Insufficient records'}

            std_dev = np.std(values)
            if std_dev == 0:
                logger.warning("All ratings are identical, cannot normalize")
                return {'error': 'No variance in ratings'}

            # Determine mean (custom or automatic)
            if midpoint_record:
                mean_val = midpoint_record.get_field_val("avg_rating", 25.0)
            else:
                mean_val = np.mean(values)

            # Store normalization parameters in rating job
            data_service = self._get_data_service()
            rating_job = data_service.get_rating_job()
            rating_job.set_field_val("transform_mean", mean_val)
            rating_job.set_field_val("transform_std_dev", std_dev)

            # Apply Z-score normalization
            normalized_records = 0
            for rec in records:
                current_rating = rec.get_field_val("avg_rating", mean_val)
                z_score = (current_rating - mean_val) / std_dev

                # Scale to 1-10 range
                scaled_rating = (((z_score - (-3)) / (3 - (-3))) * (10.0 - 1.0)) + 1.0
                scaled_rating = max(1.0, min(10.0, scaled_rating))

                if update_records:
                    rec.set_field_val("avg_rating", scaled_rating)
                    rec.set_field_val("value", round(scaled_rating, 2))

                normalized_records += 1

            result = {
                'records_normalized': normalized_records,
                'mean': mean_val,
                'std_dev': std_dev,
                'method': 'zscore_with_midpoint' if midpoint_record else 'zscore',
                'midpoint_record_id': str(midpoint_record.id) if midpoint_record else None
            }

            logger.info(f"Normalized {normalized_records} records using Z-score method")
            return result

        except Exception as e:
            logger.error(f"Error in Z-score normalization: {e}")
            return {'error': str(e)}

    def denormalize_rating(self, normalized_rating: float, transform_mean: float,
                         transform_std_dev: float) -> float:
        """
        Denormalize a rating from 1-10 scale back to TrueSkill mu scale
        Reverse operation of normalize_ratings_zscore
        """
        try:
            # Convert from 1-10 scale to Z-score
            z_score = ((normalized_rating - 1.0) / (10.0 - 1.0)) * (3 - (-3)) + (-3)

            # Convert from Z-score back to mu scale
            mu_value = z_score * transform_std_dev + transform_mean

            return mu_value

        except Exception as e:
            logger.error(f"Error denormalizing rating: {e}")
            return transform_mean

    def initialize_new_record_ratings(self, records: List[AnnotationRecord],
                                    use_predictions: bool = False) -> int:
        """Initialize ratings for new records"""
        try:
            config = self._config_service.get_config()
            data_service = self._get_data_service()

            # Calculate baseline rating
            base_rating = self.calculate_average_conservative_rating(data_service.get_manual_voted_list())

            initialized_count = 0

            for record in records:
                if use_predictions and config.get_model_sigma_config() and config.get_model_sigma_config().is_model_loaded():
                    try:
                        # Use model prediction
                        if record.file:
                            predicted_rating = config.get_model_sigma_config().predict_rating(record.file.full_path)
                            if predicted_rating is not None:
                                # Get transform parameters for denormalization
                                rating_job = data_service.get_rating_job()
                                job_mean = rating_job.get_field_val("transform_mean", config.DEFAULT_MU)
                                job_std = rating_job.get_field_val("transform_std_dev", 1.0)

                                mu = self.denormalize_rating(predicted_rating, job_mean, job_std)
                                sigma = config.get_sigma_for_new_items()
                            else:
                                # Fallback
                                mu = base_rating
                                sigma = config.get_sigma_for_new_items()
                        else:
                            # No file for prediction
                            mu = base_rating
                            sigma = config.get_sigma_for_new_items()
                    except Exception as e:
                        logger.warning(f"Prediction failed for record {record.id}: {e}")
                        mu = base_rating
                        sigma = config.get_sigma_for_new_items()
                else:
                    # Use baseline rating
                    mu = base_rating
                    sigma = config.get_sigma_for_new_items()

                # Set initial ratings
                self.update_record_rating(record, mu, sigma, update_cache=True)
                initialized_count += 1

            logger.info(f"Initialized ratings for {initialized_count} records")
            return initialized_count

        except Exception as e:
            logger.error(f"Error initializing record ratings: {e}")
            return 0

    def get_rating_statistics(self, records: List[AnnotationRecord]) -> Dict[str, float]:
        """Calculate rating statistics for analysis"""
        try:
            if not records:
                return {}

            conservative_ratings = [self.get_conservative_rating(rec) for rec in records]
            mu_values = [rec.get_field_val("avg_rating", 25.0) for rec in records]
            sigma_values = [rec.get_field_val("trueskill_sigma", 8.333) for rec in records]

            return {
                'count': len(records),
                'conservative_mean': np.mean(conservative_ratings),
                'conservative_std': np.std(conservative_ratings),
                'mu_mean': np.mean(mu_values),
                'mu_std': np.std(mu_values),
                'sigma_mean': np.mean(sigma_values),
                'sigma_std': np.std(sigma_values),
                'conservative_range': max(conservative_ratings) - min(conservative_ratings),
                'mu_range': max(mu_values) - min(mu_values)
            }

        except Exception as e:
            logger.error(f"Error calculating rating statistics: {e}")
            return {}

    def _get_true_values(self, record: AnnotationRecord) -> Tuple[float, float]:
        """Get TrueSkill mu and sigma values, with fallbacks"""
        config = self._config_service.get_config()

        # Try cache first
        try:
            cache_service = self._get_cache_service()
            return cache_service.get_trueskill_values(str(record.id))
        except Exception:
            # Fall back to record values
            mu = record.get_field_val("avg_rating", config.DEFAULT_MU)
            sigma = record.get_field_val("trueskill_sigma", config.MODEL_SIGMA)
            return mu, sigma

    def merge_records(self, primary: AnnotationRecord, secondary: AnnotationRecord) -> bool:
        """Merge two records, keeping primary and discarding secondary"""
        try:
            # Update primary to ensure it's manual
            primary.set_field_val("manual", True)

            # Transfer any relevant data from secondary (like file references)
            # Note: In current system, merging sets secondary as non-manual

            # Update cache - clear secondary cache entries
            cache_service = self._get_cache_service()
            cache_service.clear_record_caches(str(secondary.id))

            # Secondary record becomes non-manual (effectively hidden)
            secondary.set_field_val("manual", False)

            logger.info(f"Merged record {secondary.id} into {primary.id}")
            return True

        except Exception as e:
            logger.error(f"Error merging records {secondary.id} into {primary.id}: {e}")
            return False
