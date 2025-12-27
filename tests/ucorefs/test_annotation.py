"""
UCoreFS - Annotation Workflow Tests

Tests for AnnotationJob, AnnotationRecord, and AnnotationService.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId
from datetime import datetime


class TestAnnotationModels:
    """Tests for annotation models."""
    
    def test_annotation_job_initialization(self):
        """Test AnnotationJob can be initialized."""
        from src.ucorefs.annotation.models import AnnotationJob
        
        job = AnnotationJob(
            name="Test Job",
            job_type="binary",
            choices=["yes", "no"]
        )
        
        assert job.name == "Test Job"
        assert job.job_type == "binary"
        assert job.choices == ["yes", "no"]
    
    def test_annotation_job_types(self):
        """Test different job types."""
        from src.ucorefs.annotation.models import AnnotationJob
        
        # Binary
        binary_job = AnnotationJob(name="Binary", job_type="binary")
        assert binary_job.job_type == "binary"
        
        # Multiclass
        multiclass_job = AnnotationJob(name="Multi", job_type="multiclass")
        assert multiclass_job.job_type == "multiclass"
        
        # Multilabel
        multilabel_job = AnnotationJob(name="Labels", job_type="multilabel")
        assert multilabel_job.job_type == "multilabel"
    
    def test_annotation_job_progress(self):
        """Test progress calculation."""
        from src.ucorefs.annotation.models import AnnotationJob
        
        job = AnnotationJob(
            name="Test",
            total_files=100,
            annotated_count=25
        )
        
        assert job.progress_percent == 25.0
        assert job.remaining_count == 75
    
    def test_annotation_job_progress_zero(self):
        """Test progress with zero files."""
        from src.ucorefs.annotation.models import AnnotationJob
        
        job = AnnotationJob(name="Empty", total_files=0)
        
        assert job.progress_percent == 0.0
        assert job.remaining_count == 0
    
    def test_annotation_record_initialization(self):
        """Test AnnotationRecord can be initialized."""
        from src.ucorefs.annotation.models import AnnotationRecord
        
        job_id = ObjectId()
        file_id = ObjectId()
        
        record = AnnotationRecord(
            job_id=job_id,
            file_id=file_id
        )
        
        assert record.job_id == job_id
        assert record.file_id == file_id
        assert record.is_annotated == False
    
    def test_annotation_record_set_value(self):
        """Test setting annotation value."""
        from src.ucorefs.annotation.models import AnnotationRecord
        
        record = AnnotationRecord(
            job_id=ObjectId(),
            file_id=ObjectId()
        )
        
        record.set_value("yes", annotated_by="test_user")
        
        assert record.value == "yes"
        assert record.is_annotated == True
        assert record.annotated_by == "test_user"
        assert record.annotated_at is not None
    
    def test_annotation_record_clear_value(self):
        """Test clearing annotation value."""
        from src.ucorefs.annotation.models import AnnotationRecord
        
        record = AnnotationRecord(
            job_id=ObjectId(),
            file_id=ObjectId()
        )
        
        record.set_value("yes")
        record.clear_value()
        
        assert record.value is None
        assert record.is_annotated == False
        assert record.annotated_at is None


class TestAnnotationService:
    """Tests for AnnotationService."""
    
    def test_annotation_service_has_create_job(self):
        """Test AnnotationService has create_job method."""
        from src.ucorefs.annotation.service import AnnotationService
        
        assert hasattr(AnnotationService, 'create_job')
    
    def test_annotation_service_has_annotate(self):
        """Test AnnotationService has annotate method."""
        from src.ucorefs.annotation.service import AnnotationService
        
        assert hasattr(AnnotationService, 'annotate')
    
    def test_annotation_service_has_navigation(self):
        """Test AnnotationService has navigation methods."""
        from src.ucorefs.annotation.service import AnnotationService
        
        assert hasattr(AnnotationService, 'get_next_unannotated')
        assert hasattr(AnnotationService, 'get_job_progress')
    
    def test_annotation_service_has_export(self):
        """Test AnnotationService has export method."""
        from src.ucorefs.annotation.service import AnnotationService
        
        assert hasattr(AnnotationService, 'export_annotations')
    
    def test_annotation_service_has_file_management(self):
        """Test AnnotationService has file management methods."""
        from src.ucorefs.annotation.service import AnnotationService
        
        assert hasattr(AnnotationService, 'add_files_to_job')
        assert hasattr(AnnotationService, 'add_query_to_job')


class TestAnnotationPackage:
    """Tests for annotation package exports."""
    
    def test_package_exports_models(self):
        """Test annotation package exports models."""
        from src.ucorefs.annotation import AnnotationJob, AnnotationRecord
        
        assert AnnotationJob is not None
        assert AnnotationRecord is not None
    
    def test_package_exports_service(self):
        """Test annotation package exports service."""
        from src.ucorefs.annotation import AnnotationService
        
        assert AnnotationService is not None
