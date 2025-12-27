"""
Rating operations using service architecture
DEPRECATED: This module is being migrated to service-based architecture.
All operations now delegate to appropriate services following SSOT principle.
"""
import numpy as np
import random
from typing import Optional
from tqdm import tqdm
from PySide6.QtWidgets import QMessageBox, QInputDialog
from loguru import logger

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from anotation_tool import TrueSkillAnnotationRecordTools
from constants import MODEL_SIGMA, DEFAULT_MU
from services.configuration_service import ConfigurationService

# Service imports
from services.service_container import service_container
from services.rating_service import RatingService
from services.data_service import DataService
from services.validation_service import ValidationService


class RatingOperations:
    """Static class for rating data manipulation operations"""

    @staticmethod
    def calculate_mean_conservative_rating(data_manager) -> float:
        """
        DEPRECATED: Use RatingService.calculate_average_conservative_rating() instead
        Calculate mean conservative rating from existing manual voted items

        Args:
            data_manager: The data manager instance

        Returns:
            Mean conservative rating or default if no data
        """
        try:
            # Delegate to RatingService
            rating_service = service_container.get_service(RatingService)
            records = data_manager.manual_voted_list if hasattr(data_manager, 'manual_voted_list') else []
            return rating_service.calculate_average_conservative_rating(records)

        except Exception as e:
            logger.error(f"Error in service-backed calculation: {e}")
            # Fallback to original logic with proper constants
            config_service = service_container.get_service('ConfigurationService')
            config = config_service.get_config()

            if not data_manager.manual_voted_list:
                return config.DEFAULT_MU

            ratings = []
            for rec in data_manager.manual_voted_list:
                mu = rec.get_field_val("avg_rating", config.DEFAULT_MU)
                sigma = rec.get_field_val("trueskill_sigma", config.MODEL_SIGMA)
                conservative_rating = mu - 2 * sigma
                ratings.append(conservative_rating)

            if not ratings:
                return config.DEFAULT_MU

            mean_conservative = sum(ratings) / len(ratings)
            return max(mean_conservative, config.DEFAULT_MU * 0.5)

        except Exception as e:
            logger.error(f"Error calculating mean conservative rating: {e}")
            config_service = service_container.get_service('ConfigurationService')
            return config_service.get_config().DEFAULT_MU

    @staticmethod
    def normalize_ratings(parent_widget, data_manager):
        """
        DEPRECATED: Use RatingService.normalize_ratings_zscore() instead
        Normalize ratings using Z-score method

        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            # Delegate to RatingService
            rating_service = service_container.get_service(RatingService)
            records = data_manager.manual_voted_list if hasattr(data_manager, 'manual_voted_list') else []

            result = rating_service.normalize_ratings_zscore(records, update_records=True)

            if 'error' in result:
                QMessageBox.warning(parent_widget, "Error", f"Failed to normalize ratings: {result['error']}")
                return None
            else:
                logger.info(f"Normalization completed via service: {result}")
                return f"Ratings normalized (Z-score, {result.get('records_normalized', 0)} records processed)."

        except Exception as e:
            logger.error(f"Error in service-backed normalization: {e}")
            # Fallback to original implementation
            try:
                from rating_helpers import RatingHelpers
                # Original logic here with proper imports
                config_service = service_container.get_service('ConfigurationService')
                config = config_service.get_config()

                final_min, final_max = config.initial_rating_min, config.initial_rating_max
                norm_list = data_manager.manual_voted_list.copy()
                values = [ar.get_field_val("avg_rating", 0) for ar in norm_list]

                if not values:
                    QMessageBox.warning(parent_widget, "No Ratings", "No valid ratings found.")
                    return None

                mean_rating = np.mean(values)
                std_dev_rating = np.std(values)

                if std_dev_rating == 0:
                    QMessageBox.warning(parent_widget, "No Variation", "All ratings are the same.")
                    return None

                data_manager.rating_job.set_field_val("transform_mean", mean_rating)
                data_manager.rating_job.set_field_val("transform_std_dev", std_dev_rating)

                for ar in tqdm(norm_list, desc="Normalizing ratings"):
                    avg_rating = ar.get_field_val("avg_rating", mean_rating)
                    z_score = (avg_rating - mean_rating) / std_dev_rating
                    scaled_rating = (((z_score - (-3)) / (3 - (-3))) * (final_max - final_min)) + final_min
                    scaled_rating = max(final_min, min(final_max, scaled_rating))
                    ar.value = round(scaled_rating, 2)

                logger.info("Fallback normalization completed")
                return "Ratings normalized (Z-score, scaled to 1-10)."

            except Exception as fallback_error:
                logger.error(f"Fallback normalization also failed: {fallback_error}")
                QMessageBox.warning(parent_widget, "Error", f"Failed to normalize ratings: {str(e)}")
                return None

    @staticmethod
    def add_rand_from_all_to_human_voted(parent_widget, data_manager):
        """
        DEPRECATED: Service-based version should be used instead
        Add random images from all to human voted list using service architecture

        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            # Get services
            rating_service = service_container.get_service(RatingService)
            config_service = service_container.get_service(ConfigurationService)

            config = config_service.get_config()
            count = config.ADD_RAND_COUNT

            # Use existing query logic (could be moved to DataService later)
            all_list = AnnotationRecord.find({"parent_id": data_manager.rating_job.id, "manual": {"$ne": True}})

            if not all_list:
                logger.warning("No available records to add")
                QMessageBox.warning(parent_widget, "No Data", "No available images to add.")
                return None

            # Select random records
            selected_records = random.sample(all_list, min(count, len(all_list)))

            # Initialize ratings for selected records
            initialized_count = rating_service.initialize_new_record_ratings(selected_records, use_predictions=True)

            # Refresh data
            if hasattr(data_manager, 'load_manual_voted_list'):
                data_manager.load_manual_voted_list()

            logger.info(f"Added {initialized_count} random images to human voted list")
            return f"Added {initialized_count} random images to human voted list."

        except Exception as e:
            logger.error(f"Error in service-backed random addition: {e}")
            # Fallback to original implementation
            try:
                from model_config import model_sigma_config
                from rating_helpers import RatingHelpers

                config = service_container.get_service(ConfigurationService).get_config()
                count = config.ADD_RAND_COUNT
                all_list = AnnotationRecord.find({"parent_id": data_manager.rating_job.id, "manual": {"$ne": True}})

                mean_conservative_rating = RatingOperations.calculate_mean_conservative_rating(data_manager)
                sigma_to_use = model_sigma_config.get_sigma_for_new_items()
                transform_mean = data_manager.rating_job.get_field_val("transform_mean", config.DEFAULT_MU)
                transform_std_dev = data_manager.rating_job.get_field_val("transform_std_dev", 1.0)

                for _ in range(count):
                    selected_record = random.choice(all_list)
                    if model_sigma_config.use_predictions and model_sigma_config.is_model_loaded():
                        try:
                            predicted_rating = model_sigma_config.predict_rating(selected_record.file.full_path)
                            if predicted_rating is not None:
                                mu = RatingHelpers.denormalize_rating(predicted_rating, transform_mean, transform_std_dev)
                            else:
                                mu = mean_conservative_rating
                        except Exception:
                            mu = mean_conservative_rating
                    else:
                        mu = mean_conservative_rating

                    selected_record.set_field_val("manual", True)
                    selected_record.set_field_val("trueskill_sigma", sigma_to_use)
                    selected_record.set_field_val("avg_rating", mu)
                    TrueSkillAnnotationRecordTools.clear_cache(selected_record)

                data_manager.load_manual_voted_list()
                logger.info(f"Fallback added {count} random images to human voted list")
                return f"Added {count} random images to human voted list."

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                QMessageBox.warning(parent_widget, "Error", f"Failed to add random images: {str(e)}")
                return None

    @staticmethod
    def shift_all_mu_plus_10(parent_widget, data_manager):
        """Shift all manual voted items' mu by +10
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            reply = QMessageBox.question(
                parent_widget, 'Confirm Action',
                'This will add 10 points to the rating (mu) of ALL manual voted items. Are you sure?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return None

            updated_count = 0
            for rec in tqdm(data_manager.manual_voted_list, desc="Shifting mu +10"):
                current_mu = rec.get_field_val("avg_rating", DEFAULT_MU)
                new_mu = current_mu + 10.0
                data_manager.update_record_rating(rec, new_mu, rec.get_field_val("trueskill_sigma", MODEL_SIGMA))
                updated_count += 1

            logger.info(f"Shifted mu +10 for {updated_count} manual voted items")
            return f"Shifted mu +10 for {updated_count} manual voted items."

        except Exception as e:
            logger.error(f"Error shifting mu +10: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to shift mu +10: {e}")
            return None

    @staticmethod
    def keep_safe_images(parent_widget, data_manager):
        """Keep only safe images
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            query = {'parent_id': data_manager.rating_job.id, 'manual': {"$ne": None}}
            result = AnnotationRecord.find(query)
            for item in result:
                if item.file.full_path.lower().find("safe repo") == -1:
                    item.set_field_val("manual", None)
            data_manager.load_manual_voted_list()
            logger.info("Kept only safe images")
            return "Kept only safe images."

        except Exception as e:
            logger.error(f"Error keeping safe images: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to keep safe images: {e}")
            return None

    @staticmethod
    def clear_all_manual_flags(parent_widget, data_manager):
        """Clear all manual flags
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            reply = QMessageBox.question(parent_widget, 'Confirm Action',
                                         'This will remove all records from manual voted list. Are you sure?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return None

            manual_records = AnnotationRecord.find({'parent_id': data_manager.rating_job.id, 'manual': True})
            cleared_count = 0
            for rec in tqdm(manual_records, desc="Clearing manual flags"):
                rec.set_field_val("manual", None)
                cleared_count += 1

            data_manager.load_manual_voted_list()
            logger.info(f"Cleared manual flags for {cleared_count} records")
            return f"Cleared manual flags for {cleared_count} records."

        except Exception as e:
            logger.error(f"Error clearing manual flags: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to clear manual flags: {e}")
            return None

    @staticmethod
    def rem_items_with_low_sigma(parent_widget, data_manager):
        """Remove items with low sigma

        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            threshold, ok = QInputDialog.getDouble(parent_widget, "Sigma Threshold", "Enter sigma threshold:",
                                                   value=MODEL_SIGMA)
            if not ok:
                return None

            removed_count = 0
            manual_records = AnnotationRecord.find({'parent_id': data_manager.rating_job.id, 'manual': True})
            for rec in tqdm(manual_records, desc="Removing low sigma records"):
                sigma = rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                if float(sigma) > threshold:
                    rec.set_field_val("manual", None)
                    removed_count += 1
                    TrueSkillAnnotationRecordTools.clear_cache(rec)

            logger.info(f"Removed {removed_count} items with sigma > {threshold}")
            return f"Removed {removed_count} items with low sigma."

        except Exception as e:
            logger.error(f"Error removing items with low sigma: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to remove items: {e}")
            return None

    @staticmethod
    def add_1_to_all_sigma(parent_widget, data_manager):
        """Add +1 to sigma for all manual voted items

        Args:
            parent_widget: Parent widget for dialogs
            data_manager: The data manager instance
        """
        try:
            reply = QMessageBox.question(
                parent_widget, 'Confirm Action',
                'This will add 1 point to the sigma of ALL manual voted items. Are you sure?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return None

            updated_count = 0
            for rec in tqdm(data_manager.manual_voted_list, desc="Adding +1 to sigma"):
                if rec.get_field_val("ankor", False):
                    continue  # Skip anchors
                current_mu = rec.get_field_val("avg_rating", DEFAULT_MU)
                current_sigma = rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                new_sigma = current_sigma + 1.0
                data_manager.update_record_rating(rec, current_mu, new_sigma)
                updated_count += 1

            logger.info(f"Added +1 to sigma for {updated_count} manual voted items")
            return f"Added +1 to sigma for {updated_count} manual voted items."

        except Exception as e:
            logger.error(f"Error adding +1 to sigma: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to add +1 to sigma: {e}")
            return None
