"""
UExplorer - Maintenance Panel

Docking panel for viewing and controlling automated maintenance tasks.
"""
from typing import Optional, Dict, Any
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QLabel, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from loguru import logger

from src.ucorefs.services.maintenance_service import MaintenanceService
from src.core.scheduling import PeriodicTaskScheduler


class MaintenanceTaskWidget(QWidget):
    """Widget displaying a single maintenance task."""
    
    run_requested = Signal(str)  # task_name
    
    def __init__(self, task_name: str, task_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.task_name = task_name
        self.task_config = task_config
        self.last_run_time: Optional[datetime] = None
        self.status = "Idle"
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Task header
        header_layout = QHBoxLayout()
        
        # Task name (formatted)
        task_display_name = self.task_name.replace('_', ' ').title()
        name_label = QLabel(f"<b>{task_display_name}</b>")
        name_label.setFont(QFont("Segoe UI", 10))
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.status_label)
        
        layout.addLayout(header_layout)
        
        # Task info
        info_layout = QHBoxLayout()
        
        # Last run time
        self.last_run_label = QLabel("Last Run: Never")
        self.last_run_label.setStyleSheet("color: #888; font-size: 11px;")
        info_layout.addWidget(self.last_run_label)
        
        info_layout.addStretch()
        
        # Interval display
        interval_text = self._format_interval()
        interval_label = QLabel(f"Every {interval_text}")
        interval_label.setStyleSheet("color: #888; font-size: 11px;")
        info_layout.addWidget(interval_label)
        
        layout.addLayout(info_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Now")
        self.run_button.setMaximumWidth(100)
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.task_name))
        button_layout.addWidget(self.run_button)
        
        button_layout.addStretch()
        
        # Enabled/Disabled toggle (TODO: implement actual toggle)
        enabled = self.task_config.get('enabled', True)
        status_text = "Enabled" if enabled else "Disabled"
        self.enabled_label = QLabel(status_text)
        self.enabled_label.setStyleSheet(
            f"color: {'#4CAF50' if enabled else '#f44336'}; font-weight: bold;"
        )
        button_layout.addWidget(self.enabled_label)
        
        layout.addLayout(button_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #333;")
        layout.addWidget(separator)
    
    def _format_interval(self) -> str:
        """Format the interval for display."""
        minutes = self.task_config.get('interval_minutes', 0)
        hours = self.task_config.get('interval_hours', 0)
        days = self.task_config.get('interval_days', 0)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "Unknown"
    
    def update_status(self, status: str, last_run: Optional[datetime] = None):
        """Update task status display."""
        self.status = status
        
        # Update status label
        if status == "Running":
            self.status_label.setText("● Running")
            self.status_label.setStyleSheet("color: #FFC107;")
        elif status == "Completed":
            self.status_label.setText("✓ Completed")
            self.status_label.setStyleSheet("color: #4CAF50;")
        elif status == "Failed":
            self.status_label.setText("✗ Failed")
            self.status_label.setStyleSheet("color: #f44336;")
        else:
            self.status_label.setText("○ Idle")
            self.status_label.setStyleSheet("color: #888;")
        
        # Update last run time
        if last_run:
            self.last_run_time = last_run
            time_str = last_run.strftime("%H:%M:%S")
            self.last_run_label.setText(f"Last Run: {time_str}")


class MaintenancePanel(QWidget):
    """
    Docking panel for maintenance task management.
    
    Features:
    - View all scheduled maintenance tasks
    - See last run time and status for each task
    - Manually trigger tasks
    - View execution history
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.maintenance_service: Optional[MaintenanceService] = None
        self.scheduler: Optional[PeriodicTaskScheduler] = None
        self.task_widgets: Dict[str, MaintenanceTaskWidget] = {}
        
        self._init_ui()
        
        # Auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_status)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Panel header
        header = QLabel("<h2>Maintenance Tasks</h2>")
        header.setStyleSheet("color: #fff; padding: 10px;")
        layout.addWidget(header)
        
        # Scroll area for tasks
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # Container for task widgets
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setContentsMargins(0, 0, 0, 0)
        self.tasks_layout.setSpacing(5)
        
        scroll.setWidget(self.tasks_container)
        layout.addWidget(scroll)
        
        # Execution history section
        history_group = QGroupBox("Recent Executions")
        history_layout = QVBoxLayout(history_group)
        
        self.history_label = QLabel("No recent executions")
        self.history_label.setStyleSheet("color: #888; padding: 10px;")
        self.history_label.setWordWrap(True)
        history_layout.addWidget(self.history_label)
        
        layout.addWidget(history_group)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Status")
        refresh_button.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_button)
    
    def set_services(self, maintenance_service: MaintenanceService, scheduler: PeriodicTaskScheduler):
        """Set the maintenance service and scheduler instances."""
        self.maintenance_service = maintenance_service
        self.scheduler = scheduler
        
        # Load task schedule from scheduler
        if self.scheduler:
            schedule = self.scheduler._get_schedule_config()
            self._populate_tasks(schedule)
    
    def _populate_tasks(self, schedule: Dict[str, Dict[str, Any]]):
        """Populate the task list from scheduler configuration."""
        # Clear existing widgets
        for widget in self.task_widgets.values():
            widget.deleteLater()
        self.task_widgets.clear()
        
        # Create widgets for each task
        for task_name, task_config in schedule.items():
            task_widget = MaintenanceTaskWidget(task_name, task_config)
            task_widget.run_requested.connect(self._on_run_task)
            
            self.tasks_layout.addWidget(task_widget)
            self.task_widgets[task_name] = task_widget
        
        # Add stretch at the end
        self.tasks_layout.addStretch()
    
    def _on_run_task(self, task_name: str):
        """Handle manual task execution request."""
        if not self.maintenance_service:
            logger.warning("MaintenanceService not available")
            return
        
        logger.info(f"Manual execution requested for task: {task_name}")
        
        # Update UI to show running status
        if task_name in self.task_widgets:
            self.task_widgets[task_name].update_status("Running")
        
        # Execute task asynchronously
        import asyncio
        asyncio.create_task(self._execute_task(task_name))
    
    async def _execute_task(self, task_name: str):
        """Execute a maintenance task."""
        try:
            result = None
            
            # Route to appropriate method
            if task_name == 'background_verification':
                await self.maintenance_service.background_count_verification()
            elif task_name == 'database_optimization':
                result = await self.maintenance_service.database_optimization()
            elif task_name == 'cache_cleanup':
                result = await self.maintenance_service.cache_cleanup()
            elif task_name == 'orphaned_cleanup':
                result = await self.maintenance_service.cleanup_orphaned_file_records()
            elif task_name == 'log_rotation':
                result = await self.maintenance_service.log_rotation()
            elif task_name == 'database_cleanup':
                result = await self.maintenance_service.cleanup_old_records()
            
            # Update UI
            if task_name in self.task_widgets:
                self.task_widgets[task_name].update_status("Completed", datetime.now())
            
            # Update history
            self._add_history_entry(task_name, "Completed", result)
            
            logger.info(f"Task {task_name} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}")
            
            if task_name in self.task_widgets:
                self.task_widgets[task_name].update_status("Failed")
            
            self._add_history_entry(task_name, "Failed", str(e))
    
    def _add_history_entry(self, task_name: str, status: str, result: Any):
        """Add an entry to the execution history."""
        time_str = datetime.now().strftime("%H:%M:%S")
        task_display = task_name.replace('_', ' ').title()
        
        # Format result
        result_str = ""
        if isinstance(result, dict):
            # Show key metrics
            if 'duration' in result:
                result_str = f" ({result['duration']:.2f}s)"
            if 'files_deleted' in result:
                result_str += f" - {result['files_deleted']} files"
            if 'indexes_rebuilt' in result:
                result_str += f" - {result['indexes_rebuilt']} indexes"
        
        entry = f"[{time_str}] {task_display}: {status}{result_str}"
        
        # Prepend to history (most recent first)
        current_text = self.history_label.text()
        if current_text == "No recent executions":
            new_text = entry
        else:
            # Keep only last 10 entries
            entries = current_text.split('\n')
            entries.insert(0, entry)
            new_text = '\n'.join(entries[:10])
        
        self.history_label.setText(new_text)
    
    def _refresh_status(self):
        """Refresh task status display."""
        # This would query actual task status from TaskSystem
        # For now, just reset to idle if not running
        for widget in self.task_widgets.values():
            if widget.status not in ["Running"]:
                widget.update_status("Idle")
