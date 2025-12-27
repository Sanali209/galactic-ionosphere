"""
UCoreFS - Annotation Service

Service for managing annotation jobs and records.
"""
from typing import List, Optional, Any
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.annotation.models import AnnotationJob, AnnotationRecord
from src.ucorefs.models.file_record import FileRecord


class AnnotationService(BaseSystem):
    """
    Service for managing annotation workflows.
    
    Provides methods to:
    - Create and manage annotation jobs
    - Annotate files within jobs
    - Navigate unannotated files
    - Export annotated datasets
    
    Usage:
        annotation_service = locator.get_system(AnnotationService)
        
        # Create job
        job = await annotation_service.create_job(
            name="NSFW Detection",
            job_type="binary",
            choices=["safe", "nsfw"]
        )
        
        # Add files to job
        await annotation_service.add_files_to_job(job._id, file_ids)
        
        # Annotate
        await annotation_service.annotate(job._id, file_id, "safe")
        
        # Get next unannotated
        next_file = await annotation_service.get_next_unannotated(job._id)
    """
    
    async def initialize(self) -> None:
        """Initialize annotation service."""
        logger.info("AnnotationService initializing")
        await super().initialize()
        logger.info("AnnotationService ready")
    
    async def shutdown(self) -> None:
        """Shutdown annotation service."""
        logger.info("AnnotationService shutting down")
        await super().shutdown()
    
    # ==================== Job Management ====================
    
    async def create_job(
        self,
        name: str,
        job_type: str = "binary",
        choices: List[str] = None,
        description: str = "",
        created_by: str = "user"
    ) -> AnnotationJob:
        """
        Create a new annotation job.
        
        Args:
            name: Job name
            job_type: "binary", "multiclass", or "multilabel"
            choices: Available labels/choices
            description: Job description
            created_by: Creator identifier
            
        Returns:
            Created AnnotationJob
        """
        if choices is None:
            choices = ["yes", "no"] if job_type == "binary" else []
        
        job = AnnotationJob(
            name=name,
            job_type=job_type,
            choices=choices,
            description=description,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        await job.save()
        
        logger.info(f"Created annotation job: {name} ({job_type})")
        return job
    
    async def get_job(self, job_id: ObjectId) -> Optional[AnnotationJob]:
        """Get annotation job by ID."""
        return await AnnotationJob.get(job_id)
    
    async def list_jobs(self) -> List[AnnotationJob]:
        """List all annotation jobs."""
        """List all annotation jobs."""
        return await AnnotationJob.find({})
    
    async def delete_job(self, job_id: ObjectId) -> bool:
        """Delete job and all its annotation records."""
        try:
            # Delete all records for this job
            await AnnotationRecord.delete_many({"job_id": job_id})
            
            # Delete job
            job = await AnnotationJob.get(job_id)
            if job:
                await job.delete()
            
            logger.info(f"Deleted annotation job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job: {e}")
            return False
    
    # ==================== File Management ====================
    
    async def add_files_to_job(
        self,
        job_id: ObjectId,
        file_ids: List[ObjectId]
    ) -> int:
        """
        Add files to an annotation job.
        
        Creates AnnotationRecord for each file.
        
        Args:
            job_id: Job ObjectId
            file_ids: List of file ObjectIds to add
            
        Returns:
            Number of files added
        """
        job = await AnnotationJob.get(job_id)
        if not job:
            return 0
        
        added = 0
        for file_id in file_ids:
            # Check if already exists
            existing = await AnnotationRecord.find_one({
                "job_id": job_id,
                "file_id": file_id
            })
            
            if not existing:
                record = AnnotationRecord(
                    name=f"annotation_{file_id}",
                    job_id=job_id,
                    file_id=file_id,
                    is_annotated=False
                )
                await record.save()
                added += 1
        
        # Update job stats
        job.total_files += added
        await job.save()
        
        logger.info(f"Added {added} files to job {job_id}")
        return added
    
    async def add_query_to_job(
        self,
        job_id: ObjectId,
        query: dict,
        limit: int = 1000
    ) -> int:
        """
        Add files matching a query to the job.
        
        Args:
            job_id: Job ObjectId
            query: MongoDB query dict
            limit: Maximum files to add
            
        Returns:
            Number of files added
        """
        files = await FileRecord.find(query).limit(limit).to_list()
        file_ids = [f._id for f in files]
        return await self.add_files_to_job(job_id, file_ids)
    
    # ==================== Annotation ====================
    
    async def annotate(
        self,
        job_id: ObjectId,
        file_id: ObjectId,
        value: Any,
        annotated_by: str = "user",
        confidence: float = 1.0,
        notes: str = ""
    ) -> bool:
        """
        Set annotation for a file.
        
        Args:
            job_id: Job ObjectId
            file_id: File ObjectId
            value: Annotation value (string for binary/multiclass, list for multilabel)
            annotated_by: Annotator identifier
            confidence: Confidence level 0.0-1.0
            notes: Optional notes
            
        Returns:
            True if successful
        """
        record = await AnnotationRecord.find_one({
            "job_id": job_id,
            "file_id": file_id
        })
        
        if not record:
            logger.warning(f"No record found for file {file_id} in job {job_id}")
            return False
        
        # Update annotation
        was_annotated = record.is_annotated
        record.set_value(value, annotated_by)
        record.confidence = confidence
        record.notes = notes
        await record.save()
        
        # Update job stats if newly annotated
        if not was_annotated:
            job = await AnnotationJob.get(job_id)
            if job:
                job.annotated_count += 1
                await job.save()
        
        return True
    
    async def skip(
        self,
        job_id: ObjectId,
        file_id: ObjectId
    ) -> bool:
        """Mark a file as skipped."""
        record = await AnnotationRecord.find_one({
            "job_id": job_id,
            "file_id": file_id
        })
        
        if record:
            record.skipped = True
            await record.save()
            return True
        return False
    
    # ==================== Navigation ====================
    
    async def get_next_unannotated(
        self,
        job_id: ObjectId,
        skip_skipped: bool = True
    ) -> Optional[dict]:
        """
        Get next unannotated file in job.
        
        Args:
            job_id: Job ObjectId
            skip_skipped: Whether to skip files marked as skipped
            
        Returns:
            Dict with annotation_record and file, or None if complete
        """
        query = {
            "job_id": job_id,
            "is_annotated": False
        }
        
        if skip_skipped:
            query["skipped"] = {"$ne": True}
        
        record = await AnnotationRecord.find_one(query)
        
        if record:
            file = await FileRecord.get(record.file_id)
            return {
                "record": record,
                "file": file
            }
        
        return None
    
    async def get_job_progress(self, job_id: ObjectId) -> dict:
        """
        Get annotation progress for a job.
        
        Returns:
            Dict with total, annotated, skipped, remaining, percent
        """
        job = await AnnotationJob.get(job_id)
        if not job:
            return {}
        
        skipped_count = await AnnotationRecord.count({
            "job_id": job_id,
            "skipped": True
        })
        
        return {
            "total": job.total_files,
            "annotated": job.annotated_count,
            "skipped": skipped_count,
            "remaining": job.remaining_count,
            "percent": job.progress_percent
        }
    
    # ==================== Export ====================
    
    async def export_annotations(
        self,
        job_id: ObjectId,
        format: str = "json"
    ) -> List[dict]:
        """
        Export annotations for ML training.
        
        Args:
            job_id: Job ObjectId
            format: Export format ("json", "csv")
            
        Returns:
            List of annotation dicts
        """
        records = await AnnotationRecord.find({
            "job_id": job_id,
            "is_annotated": True
        }).to_list()
        
        exports = []
        for record in records:
            file = await FileRecord.get(record.file_id)
            if file:
                exports.append({
                    "file_id": str(record.file_id),
                    "file_path": file.path,
                    "file_name": file.name,
                    "value": record.value,
                    "confidence": record.confidence,
                    "annotated_at": record.annotated_at.isoformat() if record.annotated_at else None,
                    "annotated_by": record.annotated_by
                })
        
        return exports
