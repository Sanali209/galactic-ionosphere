import pytest
import numpy as np
import unittest
from unittest.mock import MagicMock
from src.core.ai.service import EmbeddingService

# Use real class but mock dependencies if possible?
# EmbeddingService is stateful (loads model).
# We want to test logic without actually loading heavy model.

def test_embedding_service_singleton():
    s1 = EmbeddingService()
    s2 = EmbeddingService()
    assert s1 is s2

def test_embedding_generation_no_model(mock_embedding_service):
    # This tests the mock basically, or if we mock internal `embedding_model`
    service = EmbeddingService()
    # Mock the internal model
    encoder = MagicMock()
    encoder.encode_image.return_value = np.zeros(512)
    service.embedding_model = encoder
    
    with unittest.mock.patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock(mode='RGB')
        vec = service.generate_embedding("dummy_path.jpg")
        
    assert len(vec) == 512
    encoder.encode_image.assert_called_once()
