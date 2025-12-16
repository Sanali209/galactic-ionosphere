import pytest
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session")
def qapp():
    """
    Ensure a QCoreApplication exists for tests that use QObjects.
    """
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        yield app
    except ImportError:
        # If PySide6 is not installed (e.g. CI), skip
        yield None
