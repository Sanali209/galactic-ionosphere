"""
Data Service - Single access point for all data operations
Following Repository pattern with centralized data management.
"""
from typing import List, Optional
from SLM.files_db.annotation_tool.annotation import AnnotationJob, AnnotationRecord
from SLM.files_db.components.File_record_wraper import get_file_record_by_folder
from loguru import logger




class DataService:
    """Unified interface for all data operations"""

    def __init__(self):
        self._annotation_job: Optional[AnnotationJob] = None
        self._all_annotations: List[AnnotationRecord] = []
        self._manual_voted_list: List[AnnotationRecord] = []
        self._is_initialized = False
        from services.configuration_service import ConfigurationService
        # Lazy load configuration service
        from services.service_container import service_container
        self._config_service = service_container.get_service(ConfigurationService)

        logger.info("DataService initialized")

    def initialize(self):
        """Initialize data connections and load basic data"""
        if self._is_initialized:
            return True

        try:
            self._ensure_rating_job()
            self._load_annotations()
            self._is_initialized = True
            logger.info("DataService fully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DataService: {e}")
            return False

    def _ensure_rating_job(self):
        """Ensure rating job exists"""
        config = self._config_service.get_config()

        if self._annotation_job is None:
            self._annotation_job = AnnotationJob.get_by_name(config.DEFAULT_JOB_NAME)
            if not self._annotation_job:
                try:
                    job_data = config.get_job_creation_data()
                    job_id = AnnotationJob.collection().insert_one(job_data).inserted_id
                    self._annotation_job = AnnotationJob(job_id)
                    logger.info(f"Created new rating job: {config.DEFAULT_JOB_NAME}")
                except Exception as e:
                    logger.error(f"Error creating new job {config.DEFAULT_JOB_NAME}: {e}")
                    raise

    def _load_annotations(self):
        """Load annotation records"""
        if not self._annotation_job:
            raise RuntimeError("Rating job not initialized")

        # Load all annotations
        query = {"parent_id": self._annotation_job.id}
        self._all_annotations = AnnotationRecord.find(query)

        # Load manual voted list
        query_manual = {"parent_id": self._annotation_job.id, "manual": True}
        self._manual_voted_list = AnnotationRecord.find(query_manual)

        logger.info(f"Loaded {len(self._all_annotations)} total, {len(self._manual_voted_list)} manual annotations")

    # Public interface methods
    def get_rating_job(self) -> AnnotationJob:
        """Get the current annotation job"""
        if not self._annotation_job:
            self.initialize()
        return self._annotation_job

    def get_all_annotations(self) -> List[AnnotationRecord]:
        """Get all annotations"""
        if not self._is_initialized:
            self.initialize()
        return self._all_annotations

    def get_manual_voted_list(self) -> List[AnnotationRecord]:
        """Get manual voted annotations"""
        if not self._is_initialized:
            self.initialize()
        return self._manual_voted_list

    def refresh_manual_voted_list(self):
        """Refresh the manual voted list from database"""
        if not self._annotation_job:
            self.initialize()

        query_manual = {"parent_id": self._annotation_job.id, "manual": True}
        self._manual_voted_list = AnnotationRecord.find(query_manual)
        logger.debug(f"Refreshed manual voted list: {len(self._manual_voted_list)} records")

    def find_annotation_by_id(self, record_id) -> Optional[AnnotationRecord]:
        """Find annotation record by ID"""
        try:
            return AnnotationRecord(record_id)
        except Exception:
            logger.warning(f"Annotation record not found: {record_id}")
            return None

    def add_files_from_folder(self, folder_path: str, count: int) -> int:
        """Add images from folder to the rating system"""
        try:
            # Ensure initialization
            if not self._is_initialized:
                self.initialize()

            # Use existing logic from data_manager but centralized here
            # This consolidates the add_folder_to_rating_job functionality
            import random
            from tqdm import tqdm

            config = self._config_service.get_config()
            folder_path = folder_path.replace("/", "\\")

            file_records = get_file_record_by_folder(folder_path, recurse=True)

            # Filter image files
            image_file_records = []
            for fr in tqdm(file_records, desc="Filtering image files"):
                if fr and fr.name and self._is_image_file(fr.name):
                    image_file_records.append(fr)

            if not image_file_records:
                logger.warning("No image files found in the folder")
                return 0

            # Process files for annotation records
            selectable_files = []
            created_count = 0

            for fr in tqdm(image_file_records, desc="Processing image files"):
                ar = self._annotation_job.get_annotation_record(fr)

                if ar:
                    if not ar.get_field_val("manual", False):
                        selectable_files.append((fr, ar))
                else:
                    try:
                        ar_data = {
                            "parent_id": self._annotation_job.id,
                            "file_id": fr.id,
                            "manual": False,
                            "avg_rating": config.DEFAULT_MU,
                            "trueskill_sigma": config.MODEL_SIGMA,
                            "value": config.DEFAULT_RATING
                        }
                        ar_id = AnnotationRecord.collection().insert_one(ar_data).inserted_id
                        ar = AnnotationRecord(ar_id)
                        selectable_files.append((fr, ar))
                        created_count += 1
                    except Exception as e:
                        logger.error(f"Error creating annotation record for {fr.name}: {e}")
                        continue

            if not selectable_files:
                return 0

            # Select random subset
            selected = random.sample(selectable_files, min(count, len(selectable_files)))

            # Mark as manual and set initial ratings
            for fr, ar in tqdm(selected, desc="Adding images to manual voting"):
                ar.set_field_val("manual", True)
                # Rating will be set by RatingService when processing

            # Refresh data
            self.refresh_manual_voted_list()

            logger.info(f"Added {len(selected)} images from folder ({created_count} new records created)")
            return len(selected)

        except Exception as e:
            logger.error(f"Error adding folder to rating system: {e}")
            raise

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is an image"""
        if not filename:
            return False
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

    # Repository pattern methods for specific operations
    def update_annotation_rating(self, annotation: AnnotationRecord, mu: float, sigma: float):
        """Update annotation record rating"""
        try:
            annotation.set_field_val("avg_rating", mu)
            annotation.set_field_val("trueskill_sigma", sigma)
            annotation.set_field_val("manual", True)
            logger.debug(f"Updated annotation {annotation.id}: μ={mu:.2f}, σ={sigma:.2f}")
        except Exception as e:
            logger.error(f"Error updating annotation rating: {e}")
            raise

    def save_annotation(self, annotation: AnnotationRecord):
        """Save annotation changes (placeholder for future batch operations)"""
        # In current implementation, changes are saved immediately via set_field_val
        # This method is for future batch save operations
        logger.debug(f"Annotation changes saved for {annotation.id}")

    def get_annotations_by_rating_range(self, min_rating: float, max_rating: float) -> List[AnnotationRecord]:
        """Get annotations within a rating range"""
        return [
            rec for rec in self._manual_voted_list
            if min_rating <= self._get_conservative_rating(rec) <= max_rating
        ]

    def _get_conservative_rating(self, record: AnnotationRecord) -> float:
        """Get conservative rating (μ - 3σ)"""
        mu = record.get_field_val("avg_rating", 25.0)
        sigma = record.get_field_val("trueskill_sigma", 8.333)
        return mu - 3 * sigma
