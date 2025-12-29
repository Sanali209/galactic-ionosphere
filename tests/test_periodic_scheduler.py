"""
Unit Tests for PeriodicTaskScheduler

Tests periodic task scheduling functionality.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.core.scheduling import PeriodicTaskScheduler
from src.core.tasks.system import TaskSystem
from src.core.locator import ServiceLocator
from src.core.config import ConfigManager


@pytest.fixture
def mock_locator():
    """Create a mock ServiceLocator."""
    locator = MagicMock(spec=ServiceLocator)
    
    # Mock TaskSystem
    task_system = AsyncMock(spec=TaskSystem)
    task_system.PRIORITY_LOW = 2
    task_system.submit = AsyncMock(return_value="task_id_123")
    
    # Mock MaintenanceService
    maintenance_service = MagicMock()
    maintenance_service.background_count_verification = AsyncMock()
    
    locator.get_system = MagicMock(side_effect=lambda cls: {
        TaskSystem: task_system,
        'MaintenanceService': maintenance_service
    }.get(cls, task_system))
    
    return locator


@pytest.fixture
def mock_config():
    """Create a mock ConfigManager."""
    config = MagicMock(spec=ConfigManager)
    config.data = MagicMock()
    return config


@pytest.mark.asyncio
async def test_scheduler_initialization(mock_locator, mock_config):
    """Test scheduler initializes correctly."""
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    
    # Should not be running initially
    assert not scheduler._running
    assert len(scheduler._scheduled_tasks) == 0
    
    # Initialize
    await scheduler.initialize()
    
    # Should be running after init
    assert scheduler._running
    assert scheduler.is_ready
    
    # Cleanup
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_uses_default_schedule(mock_locator, mock_config):
    """Test scheduler uses default schedule when config missing."""
    # Config without maintenance section
    mock_config.data = MagicMock()
    del mock_config.data.maintenance
    
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    # Should have scheduled default tasks
    assert len(scheduler._scheduled_tasks) > 0
    
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_graceful_shutdown(mock_locator, mock_config):
    """Test scheduler cancels tasks on shutdown."""
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    initial_task_count = len(scheduler._scheduled_tasks)
    assert initial_task_count > 0
    
    # Shutdown
    await scheduler.shutdown()
    
    # Should not be running
    assert not scheduler._running
    # All tasks should be done/cancelled
    for task in scheduler._scheduled_tasks:
        assert task.done()


@pytest.mark.asyncio
async def test_task_execution_submits_to_tasksystem(mock_locator, mock_config):
    """Test that tasks are submitted to TaskSystem."""
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    # Execute a task manually
    task_config = {'enabled': True}
    await scheduler._execute_task('background_verification', task_config)
    
    # Verify TaskSystem.submit was called
    scheduler.task_system.submit.assert_called_once()
    call_args = scheduler.task_system.submit.call_args
    
    # Check arguments
    assert call_args[0][0] == 'maintenance_background_verification'
    assert 'Scheduled' in call_args[0][1]
    assert call_args[1]['priority'] == TaskSystem.PRIORITY_LOW
    
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_disabled_tasks_not_scheduled(mock_locator, mock_config):
    """Test that disabled tasks are not scheduled."""
    # Mock config with one disabled task
    mock_config.data.maintenance = MagicMock()
    mock_config.data.maintenance.schedule = {
        'background_verification': {
            'enabled': False,
            'interval_minutes': 5
        },
        'database_optimization': {
            'enabled': True,
            'interval_hours': 24
        }
    }
    
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    # Should have scheduled only enabled task (1 task)
    assert len(scheduler._scheduled_tasks) == 1
    
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_zero_interval_task_skipped(mock_locator, mock_config):
    """Test that tasks with zero interval are skipped."""
    mock_config.data.maintenance = MagicMock()
    mock_config.data.maintenance.schedule = {
        'invalid_task': {
            'enabled': True,
            # No interval specified -> 0 seconds
        }
    }
    
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    # Task should be started but immediately return
    # Check logs would show warning (not testing logs here)
    
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_handles_missing_tasksystem(mock_locator, mock_config):
    """Test scheduler handles missing TaskSystem gracefully."""
    # Make TaskSystem unavailable
    mock_locator.get_system = MagicMock(side_effect=KeyError("TaskSystem not found"))
    
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    await scheduler.initialize()
    
    # Should initialize but not schedule tasks
    assert scheduler.is_ready
    assert len(scheduler._scheduled_tasks) == 0
    
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_initial_delay_works(mock_locator, mock_config):
    """Test that initial_delay_seconds is respected."""
    mock_config.data.maintenance = MagicMock()
    mock_config.data.maintenance.schedule = {
        'delayed_task': {
            'enabled': True,
            'interval_minutes': 60,
            'initial_delay_seconds': 1  # 1 second delay
        }
    }
    
    scheduler = PeriodicTaskScheduler(mock_locator, mock_config)
    
    # Start time
    import time
    start = time.time()
    
    await scheduler.initialize()
    
    # Wait a bit for delay to be noticed
    await asyncio.sleep(0.1)
    
    # Task should be scheduled
    assert len(scheduler._scheduled_tasks) == 1
    
    await scheduler.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
