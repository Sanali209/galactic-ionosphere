"""
Test for verifying circular import fix.

This test verifies that ProcessingPipeline can be imported at module level
without causing circular import errors, thanks to the protocol-based
dependency injection pattern.
"""
import pytest


def test_no_circular_import_pipeline():
    """Verify ProcessingPipeline can be imported without lazy loading."""
    # This should not raise ImportError due to circular dependencies
    from src.ucorefs.processing.pipeline import ProcessingPipeline
    assert ProcessingPipeline is not None


def test_no_circular_import_registry():
    """Verify ExtractorRegistry can be imported without issues."""
    from src.ucorefs.extractors import ExtractorRegistry
    assert ExtractorRegistry is not None


def test_protocol_implementation():
    """Verify ExtractorRegistry implements IExtractorRegistry protocol."""
    from src.ucorefs.extractors import ExtractorRegistry
    from src.ucorefs.extractors.protocols import IExtractorRegistry
    
    # Check that ExtractorRegistry satisfies the protocol
    # In Python, protocols use structural subtyping, so we just verify
    # that the required methods exist
    assert hasattr(ExtractorRegistry, 'get_for_phase')
    assert hasattr(ExtractorRegistry, 'list_registered')
    assert callable(ExtractorRegistry.get_for_phase)
    assert callable(ExtractorRegistry.list_registered)


def test_pipeline_uses_protocol():
    """Verify ProcessingPipeline uses IExtractorRegistry protocol."""
    from src.ucorefs.processing.pipeline import ProcessingPipeline
    from src.ucorefs.extractors.protocols import IExtractorRegistry
    
    # Check that the type hint is present in the module
    # (This is a compile-time check, but we can verify it exists)
    import inspect
    source = inspect.getsource(ProcessingPipeline.initialize)
    assert 'IExtractorRegistry' in source
    assert '_extractor_registry' in source
