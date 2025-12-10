import pytest
import asyncio
from unittest.mock import MagicMock
from src.core.ai.service import EmbeddingService
from src.core.ai.vector_driver import VectorDriver

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    # Mock return of generate_embedding to strict numpy array
    import numpy as np
    service.generate_embedding.return_value = np.zeros(512, dtype=np.float32)
    service.generate_text_embedding.return_value = np.zeros(512, dtype=np.float32)
    return service

@pytest.fixture
def mock_vector_driver():
    driver = MagicMock(spec=VectorDriver)
    driver.upsert_vector = MagicMock() # method is async in real usage, but MagicMock handles await if configured?
    # For async mocks, we need AsyncMock if using python 3.8+
    # But for now let's just use standard MagicMock and handle futures if needed.
    return driver
