# UExplorer Tests

This directory contains automated tests for UExplorer using pytest-qt.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_file_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run markers
pytest tests/ -m asyncio
```

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_file_model.py` - FileModel tests
- `test_file_pane.py` - FilePaneWidget tests
- `test_main_window.py` - MainWindow tests
- `test_dialogs.py` - Dialog tests (LibraryDialog, etc.)

## Fixtures

- `qapp` - Qt Application instance
- `locator` - ServiceLocator with all UCoreFS systems
- `qtbot` - pytest-qt bot for simulating user interactions

## Test Coverage

Tests cover:
- Widget initialization
- Signal/slot connections
- Data models
- User interactions
- UI state management
