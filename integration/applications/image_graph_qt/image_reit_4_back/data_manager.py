import random
from typing import Optional, List

from PySide6.QtCore import QObject, Signal
from tqdm import tqdm

from SLM.files_db.annotation_tool.annotation import AnnotationJob, AnnotationRecord
from SLM.files_db.components.File_record_wraper import get_file_record_by_folder
from constants import DEFAULT_JOB_NAME, INITIAL_RATING_MIN, INITIAL_RATING_MAX, \
    DEFAULT_MU, MODEL_SIGMA, DEFAULT_RATING
from tools import DataChangeNotification, CacheManager, DataChangeEvent
from model_config import model_sigma_config
from loguru import logger


class DataManager(QObject):
    """Centralized data management with single source of truth"""

    # Signals for data changes
    data_changed = Signal(DataChangeNotification)

    def __init__(self):
        super().__init__()
        self.cache_manager = CacheManager()
        self.rating_job: Optional[AnnotationJob] = None
        self.all_annotations: List[AnnotationRecord] = []
        self.manual_voted_list: List[AnnotationRecord] = []
        self._is_initialized = False

    def initialize(self):
        """Initialize the data manager"""
        if self._is_initialized:
            return

        self.get_or_create_rating_job()
        self.load_all_annotations()
        self.load_manual_voted_list()
        self._is_initialized = True
        logger.info("Data manager initialized")

    def get_or_create_rating_job(self) -> AnnotationJob:
        """Get or create the rating job"""
        if self.rating_job is None:
            self.rating_job = AnnotationJob.get_by_name(DEFAULT_JOB_NAME)
            if not self.rating_job:
                try:
                    job_data = {
                        "name": DEFAULT_JOB_NAME,
                        "type": "multiclass/image",
                        "choices": [str(i) for i in range(INITIAL_RATING_MIN, INITIAL_RATING_MAX + 1)]
                    }
                    job_id = AnnotationJob.collection().insert_one(job_data).inserted_id
                    self.rating_job = AnnotationJob(job_id)
                    logger.info(f"Created new rating job: {DEFAULT_JOB_NAME}")
                except Exception as e:
                    logger.error(f"Error creating new job {DEFAULT_JOB_NAME}: {e}")
                    raise
        return self.rating_job

    def load_all_annotations(self):
        """Load all annotations from database"""
        if not self.rating_job:
            raise RuntimeError("Rating job not initialized")

        query = {"parent_id": self.rating_job.id}
        self.all_annotations = AnnotationRecord.find(query)
        logger.info(f"Loaded {len(self.all_annotations)} annotations for job '{self.rating_job.name}'")

    def load_manual_voted_list(self):
        """Load manual voted annotations"""
        if not self.rating_job:
            raise RuntimeError("Rating job not initialized")

        query = {"parent_id": self.rating_job.id, "manual": True}
        self.manual_voted_list = AnnotationRecord.find(query)
        logger.info(f"Loaded {len(self.manual_voted_list)} manual voted annotations for job '{self.rating_job.name}'")

    def update_record_rating(self, record: AnnotationRecord, mu: float, sigma: float):
        """Update a record's rating with proper cache management
        
        """
        try:

            # Update the record
            record.set_field_val("avg_rating", mu)
            record.set_field_val("trueskill_sigma", sigma)
            record.set_field_val("manual", True)

            # Update caches
            self.cache_manager.set_trueskill_values(str(record.id), mu, sigma)
            self.cache_manager.set_item_rating(str(record.id), mu, sigma)

            # Notify listeners
            self.data_changed.emit(DataChangeNotification(
                event_type=DataChangeEvent.RATING_UPDATED,
                record_id=str(record.id),
                data={"mu": mu, "sigma": sigma}
            ))

            logger.debug(f"Updated rating for record {record.id}: μ={mu:.2f}, σ={sigma:.2f}")

        except Exception as e:
            logger.error(f"Error updating record rating: {e}")
            raise

    def calculate_mean_conservative_rating(self) -> float:
        """Calculate mean conservative rating from existing manual voted items"""
        try:
            if not self.manual_voted_list:
                return DEFAULT_MU

            # Get all ratings from manual voted items
            ratings = []
            for rec in self.manual_voted_list:
                mu = rec.get_field_val("avg_rating", DEFAULT_MU)
                sigma = rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                conservative_rating = mu - 3 * sigma
                ratings.append(conservative_rating)

            if not ratings:
                return DEFAULT_MU

            # Calculate mean of conservative ratings
            mean_conservative = sum(ratings) / len(ratings)

            # Ensure the result is reasonable (not too low)
            return max(mean_conservative, DEFAULT_MU * 0.5)  # At least half of default

        except Exception as e:
            logger.error(f"Error calculating mean conservative rating: {e}")
            return DEFAULT_MU

    def add_folder_to_rating_job(self, folder_path: str, count: int) -> int:
        """Add images from folder to rating job"""
        try:
            # Ensure rating job is initialized
            if not self.rating_job:
                logger.warning("Rating job not initialized, initializing now...")
                self.initialize()
            
            if not self.rating_job:
                raise RuntimeError("Failed to initialize rating job")
            
            # Replace path separators
            folder_path = folder_path.replace("/", "\\")
            file_records = get_file_record_by_folder(folder_path, recurse=True)

            def _is_image_file(filename: str) -> bool:
                if not filename:
                    return False
                return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

            # Filter to only image files first
            image_file_records = []
            for fr in tqdm(file_records, desc="Filtering image files"):
                if fr and fr.name and _is_image_file(fr.name):
                    image_file_records.append(fr)

            if not image_file_records:
                logger.warning("No image files found in the folder")
                return 0

            # Process files: get existing annotation records or create new ones
            selectable_files = []
            created_count = 0

            for fr in tqdm(image_file_records, desc="Processing image files"):
                # Try to get existing annotation record
                ar = self.rating_job.get_annotation_record(fr)

                if ar:
                    # Existing record - only add if not already manual voted
                    if not ar.get_field_val("manual", False):
                        selectable_files.append((fr, ar))
                else:
                    # No annotation record exists - create a new one
                    try:
                        # Create new annotation record for this file
                        ar_data = {
                            "parent_id": self.rating_job.id,
                            "file_id": fr.id,
                            "manual": False,
                            "avg_rating": DEFAULT_MU,
                            "trueskill_sigma": MODEL_SIGMA,
                            "value": DEFAULT_RATING
                        }
                        ar_id = AnnotationRecord.collection().insert_one(ar_data).inserted_id
                        ar = AnnotationRecord(ar_id)
                        selectable_files.append((fr, ar))
                        created_count += 1
                        logger.debug(f"Created new annotation record for {fr.name}")
                    except Exception as e:
                        logger.error(f"Error creating annotation record for {fr.name}: {e}")
                        continue

            if not selectable_files:
                logger.warning("No suitable image files found (all may already be manually voted)")
                return 0

            # Randomly select the specified count
            selected = random.sample(selectable_files, min(count, len(selectable_files)))

            added_count = 0
            # Calculate mean conservative rating for new items
            mean_conservative_rating = self.calculate_mean_conservative_rating()
            
            # Get sigma value from model config or use default
            sigma_to_use = model_sigma_config.get_sigma_for_new_items()
            logger.info(f"Using sigma value: {sigma_to_use:.4f} (from {'model' if model_sigma_config.use_llm_sigma else 'default'})")
            
            # Get normalization params for denormalization
            transform_mean = self.rating_job.get_field_val("transform_mean", DEFAULT_MU)
            transform_std_dev = self.rating_job.get_field_val("transform_std_dev", 1.0)

            for fr, ar in tqdm(selected, desc="Adding images to manual voting"):
                # Try to predict rating if enabled
                if model_sigma_config.use_predictions and model_sigma_config.is_model_loaded():
                    try:
                        from rating_helpers import RatingHelpers
                        predicted_rating = model_sigma_config.predict_rating(fr.full_path)
                        
                        if predicted_rating is not None:
                            # Denormalize predicted rating to mu
                            mu = RatingHelpers.denormalize_rating(
                                predicted_rating, 
                                transform_mean, 
                                transform_std_dev
                            )
                            sigma = sigma_to_use
                            logger.debug(f"Using prediction: rating={predicted_rating:.2f}, mu={mu:.2f}")
                        else:
                            # Fallback
                            mu = mean_conservative_rating
                            sigma = sigma_to_use
                            logger.debug(f"Prediction failed, using fallback: mu={mu:.2f}")
                    except Exception as e:
                        logger.warning(f"Prediction error for {fr.name}, using fallback: {e}")
                        mu = mean_conservative_rating
                        sigma = sigma_to_use
                else:
                    # Current behavior (no predictions)
                    mu = mean_conservative_rating
                    sigma = sigma_to_use
                
                ar.set_field_val("manual", True)
                ar.set_field_val("trueskill_sigma", sigma)
                ar.set_field_val("avg_rating", mu)

                # Clear caches for this record
                self.cache_manager.clear_record_caches(str(ar.id))
                added_count += 1

            # Refresh data
            self.load_manual_voted_list()

            # Notify about data changes
            self.data_changed.emit(DataChangeNotification(
                event_type=DataChangeEvent.RECORD_ADDED,
                data={"count": added_count, "created_new": created_count}
            ))

            logger.info(f"Added {added_count} images from folder ({created_count} new annotation records created)")
            return added_count

        except Exception as e:
            logger.error(f"Error adding folder to rating job: {e}")
            raise

    def get_record_trueskill_key(self, record: AnnotationRecord) -> float:
        """Get TrueSkill key for sorting records"""
        try:
            mu, sigma = self.cache_manager.get_trueskill_values(str(record.id))
            return mu - 3 * sigma
        except Exception as e:
            logger.error(f"Error getting TrueSkill key for record {record.id}: {e}")
            return DEFAULT_MU

    def reset_all_sigma(self, anchors_update) -> int:
        """Reset sigma for non-anchor records to MODEL_SIGMA"""
        try:
            if not self.manual_voted_list:
                logger.warning("No manual voted records found")
                return 0

            updated_count = 0
            for rec in tqdm(self.manual_voted_list, desc="Resetting sigma (excluding anchors)"):
                # Skip anchor records
                if anchors_update and rec.get_field_val("ankor", False):
                    continue

                current_mu = rec.get_field_val("avg_rating", DEFAULT_MU)
                self.update_record_rating(rec, current_mu, MODEL_SIGMA)
                updated_count += 1

            # Notify about data changes
            self.data_changed.emit(DataChangeNotification(
                event_type=DataChangeEvent.RATING_UPDATED,
                data={"action": "reset_sigma_without_anchors", "count": updated_count}
            ))

            logger.info(f"Reset sigma for {updated_count} non-anchor records")
            return updated_count

        except Exception as e:
            logger.error(f"Error resetting sigma without anchors: {e}")
            raise

    def add_one_to_sigma_without_anchors(self) -> int:
        """Add 1.0 to sigma for non-anchor records"""
        try:
            if not self.manual_voted_list:
                logger.warning("No manual voted records found")
                return 0

            updated_count = 0
            for rec in tqdm(self.manual_voted_list, desc="Adding +1 to sigma (excluding anchors)"):
                # Skip anchor records
                if rec.get_field_val("ankor", False):
                    continue

                current_mu = rec.get_field_val("avg_rating", DEFAULT_MU)
                current_sigma = rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                new_sigma = current_sigma + 1.0

                self.update_record_rating(rec, current_mu, new_sigma)
                updated_count += 1

            # Notify about data changes
            self.data_changed.emit(DataChangeNotification(
                event_type=DataChangeEvent.RATING_UPDATED,
                data={"action": "add_one_to_sigma_without_anchors", "count": updated_count}
            ))

            logger.info(f"Added +1 to sigma for {updated_count} non-anchor records")
            return updated_count

        except Exception as e:
            logger.error(f"Error adding +1 to sigma without anchors: {e}")
            raise

    def clear_all_caches(self):
        """Clear all caches"""
        self.cache_manager.clear_all_caches()
        self.data_changed.emit(DataChangeNotification(
            event_type=DataChangeEvent.CACHE_CLEARED
        ))

    def clean_cache_duplicates(self):
        """Clean duplicate and orphaned entries from all caches
        
        Returns:
            Dictionary with statistics about cleaned entries
        """
        try:
            # Get list of valid record IDs from manual voted list
            valid_record_ids = [str(rec.id) for rec in self.manual_voted_list if rec and rec.id]

            if not valid_record_ids:
                logger.warning("No valid records found to clean cache")
                return {
                    'orphaned_pairs': 0,
                    'inconsistent_pairs': 0,
                    'orphaned_trueskill': 0,
                    'orphaned_items': 0,
                    'total_cleaned': 0
                }

            # Clean duplicates using cache manager
            stats = self.cache_manager.clean_cache_duplicates(valid_record_ids)

            # Notify about data changes
            self.data_changed.emit(DataChangeNotification(
                event_type=DataChangeEvent.CACHE_CLEARED,
                data={'action': 'clean_duplicates', 'stats': stats}
            ))

            logger.info(f"Cache duplicates cleaned: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error cleaning cache duplicates: {e}")
            raise


# Global data manager instance
data_manager = DataManager()
