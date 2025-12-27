"""
Validation Service - Unified validation system following Single Source of Truth
Centralizes all validation logic for data integrity, UI state, and business rules.
"""

from typing import List, Dict, Any, Optional, Tuple
from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from loguru import logger


class ValidationService:
    """Unified validation services for the application"""

    def __init__(self):
        # Lazy load services
        from services.service_container import service_container
        self._config_service = service_container.get_service('ConfigurationService')
        self._data_service = None
        self._rating_service = None

        logger.info("ValidationService initialized")

    def _get_data_service(self):
        """Lazy load data service"""
        if self._data_service is None:
            from services.service_container import service_container
            self._data_service = service_container.get_service('DataService')
        return self._data_service

    def _get_rating_service(self):
        """Lazy load rating service"""
        if self._rating_service is None:
            from services.service_container import service_container
            self._rating_service = service_container.get_service('RatingService')
        return self._rating_service

    # Rating validation
    def validate_rating_value(self, rating: float) -> Tuple[bool, Optional[str]]:
        """Validate a single rating value"""
        try:
            config = self._config_service.get_config()

            if not isinstance(rating, (int, float)):
                return False, "Rating must be a number"

            if rating < config.initial_rating_min:
                return False, f"Rating must be at least {config.initial_rating_min}"

            if rating > config.initial_rating_max:
                return False, f"Rating cannot exceed {config.initial_rating_max}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating rating {rating}: {e}")
            return False, f"Validation error: {str(e)}"

    def validate_rating_range(self, min_rating: float, max_rating: float) -> Tuple[bool, Optional[str]]:
        """Validate a rating range"""
        try:
            if min_rating >= max_rating:
                return False, "Minimum rating must be less than maximum rating"

            # Validate both endpoints
            for rating in [min_rating, max_rating]:
                is_valid, error_msg = self.validate_rating_value(rating)
                if not is_valid:
                    return False, error_msg

            return True, None

        except Exception as e:
            logger.error(f"Error validating rating range [{min_rating}, {max_rating}]: {e}")
            return False, f"Range validation error: {str(e)}"

    def validate_trueskill_parameters(self, mu: float, sigma: float) -> Tuple[bool, Optional[str]]:
        """Validate TrueSkill parameters"""
        try:
            if not isinstance(mu, (int, float)) or not isinstance(sigma, (int, float)):
                return False, "Mu and sigma must be numbers"

            if sigma <= 0:
                return False, "Sigma must be positive"

            # Reasonable bounds (can be adjusted)
            if mu < -1000 or mu > 1000:
                return False, "Mu value seems unreasonable (expected range: -1000 to 1000)"

            if sigma > 100:
                return False, "Sigma value seems unreasonably large (expected < 100)"

            return True, None

        except Exception as e:
            logger.error(f"Error validating TrueSkill params mu={mu}, sigma={sigma}: {e}")
            return False, f"TrueSkill validation error: {str(e)}"

    # Record validation
    def validate_annotation_record(self, record: AnnotationRecord) -> Tuple[bool, List[str]]:
        """Validate an annotation record comprehensively"""
        errors = []

        try:
            # Check basic fields
            if not hasattr(record, 'id') or not record.id:
                errors.append("Record missing valid ID")

            if not hasattr(record, 'file') or not record.file:
                errors.append("Record missing file reference")
            elif not record.file.name:
                errors.append("Record file missing name")

            # Validate rating values
            mu = record.get_field_val("avg_rating", None)
            if mu is not None:
                is_valid, error_msg = self.validate_trueskill_parameters(mu, 1.0)
                if not is_valid:
                    errors.append(f"Invalid avg_rating: {error_msg}")
            else:
                errors.append("Missing avg_rating field")

            sigma = record.get_field_val("trueskill_sigma", None)
            if sigma is not None:
                is_valid, error_msg = self.validate_trueskill_parameters(1.0, sigma)
                if not is_valid:
                    errors.append(f"Invalid trueskill_sigma: {error_msg}")
            else:
                # Allow missing sigma if mu exists (will use default)
                pass

            # Validate rating consistency
            if mu is not None and sigma is not None:
                rating_service = self._get_rating_service()
                conservative_rating = rating_service.get_conservative_rating(record)
                is_valid, error_msg = self.validate_rating_value(conservative_rating)
                if not is_valid:
                    errors.append(f"Inconsistent rating values: {error_msg}")

        except Exception as e:
            logger.error(f"Error validating annotation record {getattr(record, 'id', 'unknown')}: {e}")
            errors.append(f"Validation processing error: {str(e)}")

        return len(errors) == 0, errors

    def validate_file_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate file path for image processing"""
        try:
            import os

            if not file_path or not isinstance(file_path, str):
                return False, "File path must be a non-empty string"

            if not os.path.exists(file_path):
                return False, "File does not exist"

            if not os.path.isfile(file_path):
                return False, "Path is not a file"

            # Check if it's an image file
            _, ext = os.path.splitext(file_path.lower())
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

            if ext not in valid_extensions:
                return False, f"Unsupported file format '{ext}'. Supported: {', '.join(valid_extensions)}"

            # Check file size (optional, prevent extremely large files)
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb > 100:  # 100MB limit
                    return False, f"File too large ({size_mb:.1f}MB). Maximum allowed: 100MB"
            except (OSError, IOError):
                # Size check failed, but don't fail validation for this
                pass

            return True, None

        except Exception as e:
            logger.error(f"Error validating file path '{file_path}': {e}")
            return False, f"File validation error: {str(e)}"

    # UI State validation
    def validate_comparison_selection(self, record1: Optional[AnnotationRecord],
                                   record2: Optional[AnnotationRecord]) -> Tuple[bool, Optional[str]]:
        """Validate comparison pair selection"""
        try:
            if record1 is None or record2 is None:
                return False, "Both records must be selected"

            if record1.id == record2.id:
                return False, "Cannot compare record with itself"

            # Validate records exist
            data_service = self._get_data_service()
            if not data_service.find_annotation_by_id(str(record1.id)):
                return False, "First record not found in database"

            if not data_service.find_annotation_by_id(str(record2.id)):
                return False, "Second record not found in database"

            return True, None

        except Exception as e:
            logger.error(f"Error validating comparison selection: {e}")
            return False, f"Selection validation error: {str(e)}"

    def validate_folder_for_import(self, folder_path: str, max_files: int = 1000) -> Tuple[bool, Optional[str]]:
        """Validate folder for image import"""
        try:
            import os

            if not os.path.exists(folder_path):
                return False, "Folder does not exist"

            if not os.path.isdir(folder_path):
                return False, "Path is not a directory"

            # Count image files
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if self._is_image_file(os.path.join(root, file)):
                        image_files.append(file)
                        if len(image_files) >= max_files:
                            return False, f"Too many image files. Maximum allowed: {max_files}"

            if not image_files:
                return False, "No image files found in folder"

            return True, None

        except Exception as e:
            logger.error(f"Error validating folder '{folder_path}': {e}")
            return False, f"Folder validation error: {str(e)}"

    # Business rule validation
    def validate_normalization_parameters(self, records: List[AnnotationRecord],
                                       method: str = 'zscore') -> Tuple[bool, Optional[str]]:
        """Validate parameters for rating normalization"""
        try:
            if not records or len(records) < 2:
                return False, "At least 2 records required for normalization"

            rating_service = self._get_rating_service()

            if method == 'zscore':
                # Check for variance
                ratings = [rec.get_field_val("avg_rating", 25.0) for rec in records]
                unique_ratings = set(round(r, 6) for r in ratings)  # Round to avoid floating point precision issues

                if len(unique_ratings) < 2:
                    return False, "All records have identical ratings, cannot perform Z-score normalization"
            else:
                return False, f"Unsupported normalization method: {method}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating normalization parameters: {e}")
            return False, f"Normalization validation error: {str(e)}"

    def validate_merge_operation(self, primary: AnnotationRecord,
                               secondary: AnnotationRecord) -> Tuple[bool, Optional[str]]:
        """Validate record merge operation"""
        try:
            if primary is None or secondary is None:
                return False, "Both primary and secondary records must be provided"

            if primary.id == secondary.id:
                return False, "Cannot merge record with itself"

            # Check if both records exist
            data_service = self._get_data_service()
            if not data_service.find_annotation_by_id(str(primary.id)):
                return False, "Primary record not found"

            if not data_service.find_annotation_by_id(str(secondary.id)):
                return False, "Secondary record not found"

            return True, None

        except Exception as e:
            logger.error(f"Error validating merge operation: {e}")
            return False, f"Merge validation error: {str(e)}"

    # Batch validation
    def validate_record_batch(self, records: List[AnnotationRecord]) -> Dict[str, Any]:
        """Validate a batch of records and return comprehensive results"""
        results = {
            'total_records': len(records),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': [],
            'warnings': [],
            'success_rate': 0.0
        }

        try:
            for record in records:
                is_valid, errors = self.validate_annotation_record(record)
                if is_valid:
                    results['valid_records'] += 1
                else:
                    results['invalid_records'] += 1
                    results['errors'].extend(errors)

            if records:
                results['success_rate'] = (results['valid_records'] / len(records)) * 100

            logger.info(f"Batch validation completed: {results['valid_records']}/{results['total_records']} valid")

        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            results['errors'].append(f"Batch validation failed: {str(e)}")

        return results

    def validate_system_health(self) -> Dict[str, Any]:
        """Validate overall system health"""
        health_status = {
            'overall_status': 'unknown',
            'services_status': {},
            'data_integrity': {},
            'configuration_status': {}
        }

        try:
            # Check services
            from services.service_container import service_container
            health_status['services_status'] = service_container.get_registered_services()

            # Check configuration
            is_config_valid = self._config_service.validate_configuration()
            health_status['configuration_status'] = {
                'is_valid': is_config_valid,
                'message': 'Configuration valid' if is_config_valid else 'Configuration invalid'
            }

            # Check data integrity (sample check)
            data_service = self._get_data_service()
            try:
                records = data_service.get_manual_voted_list()
                batch_results = self.validate_record_batch(records[:10])  # Check first 10 records

                health_status['data_integrity'] = {
                    'sample_check_passed': batch_results['success_rate'] == 100.0,
                    'sample_size_checked': len(records[:10]),
                    'errors_found': len(batch_results['errors'])
                }
            except Exception as e:
                health_status['data_integrity'] = {
                    'error': f'Data integrity check failed: {str(e)}'
                }

            # Overall status
            all_healthy = (
                health_status['configuration_status']['is_valid'] and
                not health_status['data_integrity'].get('error')
            )
            health_status['overall_status'] = 'healthy' if all_healthy else 'unhealthy'

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            health_status['overall_status'] = 'error'
            health_status['error_message'] = str(e)

        return health_status

    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image (helper method)"""
        if not file_path:
            return False

        import os
        _, ext = os.path.splitext(file_path.lower())
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        return ext in valid_extensions

    # Configuration validation
    def validate_model_configuration(self) -> Tuple[bool, List[str]]:
        """Validate model configuration for predictions"""
        errors = []

        try:
            config = self._config_service.get_config()
            model_config = config.get_model_sigma_config()

            if model_config and model_config.use_predictions:
                if not model_config.is_model_loaded():
                    errors.append("Predictions enabled but model not loaded")

                if not model_config.use_llm_sigma:
                    errors.append("Using predictions but not using model sigma (inconsistent)")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating model configuration: {e}")
            return False, [f"Model validation error: {str(e)}"]

    def get_validation_report(self, records: List[AnnotationRecord]) -> str:
        """Generate a human-readable validation report"""
        try:
            results = self.validate_record_batch(records)

            report = f"""
Validation Report
=================
Total Records: {results['total_records']}
Valid Records: {results['valid_records']}
Invalid Records: {results['invalid_records']}
Success Rate: {results['success_rate']:.1f}%

Errors Found:
"""

            for error in results['errors'][:20]:  # Limit to first 20 errors
                report += f"- {error}\n"

            if len(results['errors']) > 20:
                report += f"- ... and {len(results['errors']) - 20} more errors\n"

            return report.strip()

        except Exception as e:
            return f"Error generating validation report: {str(e)}"
