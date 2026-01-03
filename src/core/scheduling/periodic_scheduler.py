"""
Core - Periodic Task Scheduler

Lightweight periodic task scheduler for automated maintenance operations.
All tasks are submitted via TaskSystem for consistency and crash recovery.
"""
from typing import Dict, Any, List
import asyncio
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.tasks.system import TaskSystem


class PeriodicTaskScheduler(BaseSystem):
    """
    Periodic task scheduler for automated maintenance.
    
    Supports interval-based scheduling (minutes, hours, days) and submits
    all tasks via TaskSystem for crash recovery and progress tracking.
    
    Usage:
        - Configure tasks in config.json under maintenance.schedule
        - Tasks are auto-scheduled on initialize()
        - All tasks submit to TaskSystem with LOW priority
    
    Example config:
        {
          "maintenance": {
            "schedule": {
              "background_verification": {
                "enabled": true,
                "interval_minutes": 5
              }
            }
          }
        }
    """
    
    # Required dependencies for task submission
    depends_on = ["TaskSystem", "MaintenanceService", "DatabaseManager"]
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._running = False
        self._scheduled_tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize scheduler and start scheduled tasks."""
        logger.info("PeriodicTaskScheduler initializing...")

        
        # Get required dependencies (guaranteed by depends_on)
        self.task_system = self.locator.get_system(TaskSystem)
        
        from src.ucorefs.services.maintenance_service import MaintenanceService
        self.maintenance_service = self.locator.get_system(MaintenanceService)
        
        # Read schedule from config
        schedule = self._get_schedule_config()
        
        if not schedule:
            logger.info("No maintenance tasks configured")
            await super().initialize()
            return
        
        # Start each scheduled task
        for task_name, task_config in schedule.items():
            if task_config.get('enabled', True):
                task_coroutine = asyncio.create_task(
                    self._run_periodic_task(task_name, task_config)
                )
                self._scheduled_tasks.append(task_coroutine)
                logger.info(f"Scheduled periodic task: {task_name}")
        
        self._running = True
        await super().initialize()
        logger.info(f"PeriodicTaskScheduler ready ({len(self._scheduled_tasks)} tasks)")
    
    async def shutdown(self):
        """Shutdown scheduler and cancel all scheduled tasks."""
        logger.info("PeriodicTaskScheduler shutting down...")
        self._running = False
        
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for graceful cancellation
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
            logger.info(f"Cancelled {len(self._scheduled_tasks)} scheduled tasks")
        
        await super().shutdown()
    
    def _get_schedule_config(self) -> Dict[str, Any]:
        """
        Read maintenance schedule from config.
        
        Returns:
            Dict of task_name -> task_config, or empty dict if not configured
        """
        # Default schedule (used if config missing)
        default_schedule = {
            'background_verification': {
                'enabled': True,
                'interval_minutes': 5
            },
            'database_optimization': {
                'enabled': True,
                'interval_hours': 24,
                'initial_delay_seconds': 300
            },
            'cache_cleanup': {
                'enabled': True,
                'interval_hours': 6
            },
            'orphaned_cleanup': {
                'enabled': True,
                'interval_hours': 12
            },
            'log_rotation': {
                'enabled': True,
                'interval_hours': 24
            },
            'database_cleanup': {
                'enabled': True,
                'interval_days': 7
            }
        }
        
        # Try to load from config
        try:
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'maintenance'):
                maintenance_config = self.config.data.maintenance
                if hasattr(maintenance_config, 'schedule'):
                    return maintenance_config.schedule
        except Exception as e:
            logger.debug(f"Failed to load maintenance schedule from config: {e}")
        
        # Return defaults
        logger.info("Using default maintenance schedule")
        return default_schedule
    
    async def _run_periodic_task(self, task_name: str, config: Dict[str, Any]):
        """
        Run a periodic task on configured interval.
        
        Args:
            task_name: Name of the task (e.g., 'background_verification')
            config: Task configuration dict with interval and other settings
        """
        # Calculate interval in seconds
        interval_seconds = (
            config.get('interval_minutes', 0) * 60 +
            config.get('interval_hours', 0) * 3600 +
            config.get('interval_days', 0) * 86400
        )
        
        if interval_seconds == 0:
            logger.warning(f"Task {task_name} has zero interval, skipping")
            return
        
        # Initial delay (optional, to stagger task starts)
        initial_delay = config.get('initial_delay_seconds', 0)
        if initial_delay > 0:
            logger.info(f"Task {task_name}: Initial delay {initial_delay}s")
            try:
                await asyncio.sleep(initial_delay)
            except asyncio.CancelledError:
                logger.info(f"Task {task_name} cancelled during initial delay")
                return
        
        logger.info(f"Starting periodic task: {task_name} (interval: {interval_seconds}s)")
        
        # Main loop
        while self._running:
            try:
                # Execute task
                await self._execute_task(task_name, config)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"Periodic task cancelled: {task_name}")
                break
            except Exception as e:
                logger.error(f"Error in periodic task {task_name}: {e}", exc_info=True)
                # Continue running even if task fails
                try:
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
    
    async def _execute_task(self, task_name: str, config: Dict[str, Any]):
        """
        Execute a single maintenance task via TaskSystem.
        
        Args:
            task_name: Name of the task to execute
            config: Task configuration dict
        """
        if not self.task_system:
            logger.warning(f"Cannot execute {task_name}: TaskSystem not available")
            return
        
        # All tasks submit to TaskSystem with LOW priority
        try:
            await self.task_system.submit(
                f"maintenance_{task_name}",
                f"Scheduled: {task_name}",
                priority=TaskSystem.PRIORITY_LOW
            )
            logger.debug(f"Submitted {task_name} to TaskSystem")
        except Exception as e:
            logger.error(f"Failed to submit {task_name} to TaskSystem: {e}")
