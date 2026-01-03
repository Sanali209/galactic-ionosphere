"""
Background Processing Panel - Real-time task monitoring.

Displays TaskSystem queue, active tasks, and processing status.
"""
from typing import TYPE_CHECKING, List, Optional, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QProgressBar, QHeaderView, QGroupBox, QPushButton
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
from loguru import logger

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator
    from src.core.tasks.system import TaskSystem
    from src.ucorefs.processing.pipeline import ProcessingPipeline


class BackgroundPanel(QWidget):
    """
    Dockable panel showing background task status.
    
    Features:
    - Active tasks table with progress
    - Queue counts
    - Recent completed tasks
    - Auto-refresh via QTimer
    """
    
    REFRESH_INTERVAL_MS = 2000  # 2 seconds
    
    def __init__(self, locator: "ServiceLocator", parent: Optional[QWidget] = None) -> None:
        """
        Initialize BackgroundPanel.
        
        Args:
            locator: ServiceLocator for accessing TaskSystem
            parent: Parent widget
        """
        super().__init__(parent)
        self.locator: "ServiceLocator" = locator
        self._task_system: Optional["TaskSystem"] = None
        self._pipeline: Optional["ProcessingPipeline"] = None
        
        self._setup_ui()
        self._setup_timer()
        
        logger.info("BackgroundPanel initialized")
    
    def _setup_ui(self):
        """Create panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # --- Queue Stats ---
        stats_group = QGroupBox("Processing Queue")
        stats_layout = QHBoxLayout(stats_group)
        
        self.workers_label = QLabel("Workers: 0")
        self.phase2_label = QLabel("Phase 2: 0")
        self.phase3_label = QLabel("Phase 3: 0")
        
        stats_layout.addWidget(self.workers_label)
        stats_layout.addWidget(self.phase2_label)
        stats_layout.addWidget(self.phase3_label)
        stats_layout.addStretch()
        
        layout.addWidget(stats_group)
        
        # --- Active Tasks Table ---
        active_group = QGroupBox("Active Tasks")
        active_layout = QVBoxLayout(active_group)
        
        self.active_table = QTableWidget()
        self.active_table.setColumnCount(4)
        self.active_table.setHorizontalHeaderLabels(["Task", "Handler", "Status", "Progress"])
        self.active_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.active_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.active_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.active_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.active_table.setColumnWidth(3, 100)
        self.active_table.setAlternatingRowColors(True)
        self.active_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.active_table.verticalHeader().setVisible(False)
        
        active_layout.addWidget(self.active_table)
        layout.addWidget(active_group)
        
        # --- Recent Completed ---
        recent_group = QGroupBox("Recent Completed")
        recent_layout = QVBoxLayout(recent_group)
        
        self.recent_table = QTableWidget()
        self.recent_table.setColumnCount(3)
        self.recent_table.setHorizontalHeaderLabels(["Task", "Handler", "Result"])
        self.recent_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.recent_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.recent_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.recent_table.setAlternatingRowColors(True)
        self.recent_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setMaximumHeight(150)
        
        recent_layout.addWidget(self.recent_table)
        layout.addWidget(recent_group)
        
        # --- Refresh Button ---
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self._refresh)
        btn_layout.addStretch()
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)
    
    def _setup_timer(self):
        """Setup auto-refresh timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self.REFRESH_INTERVAL_MS)
    
    def showEvent(self, event):
        """Start timer when shown."""
        super().showEvent(event)
        self._timer.start(self.REFRESH_INTERVAL_MS)
        self._refresh()
    
    def hideEvent(self, event):
        """Stop timer when hidden."""
        super().hideEvent(event)
        self._timer.stop()
    
    def _get_systems(self):
        """Get TaskSystem and ProcessingPipeline."""
        if not self._task_system:
            try:
                from src.core.tasks.system import TaskSystem
                self._task_system = self.locator.get_system(TaskSystem)
            except Exception:
                pass
        
        if not self._pipeline:
            try:
                from src.ucorefs.processing.pipeline import ProcessingPipeline
                self._pipeline = self.locator.get_system(ProcessingPipeline)
            except Exception:
                pass
    
    def _refresh(self):
        """Refresh task data from database."""
        self._get_systems()
        
        # Guard against overlapping refreshes (re-entrancy protection)
        if hasattr(self, '_refresh_task') and self._refresh_task and not self._refresh_task.done():
            return
        
        # Guard against modal dialog conflicts (qasync restriction)
        from PySide6.QtWidgets import QApplication
        if QApplication.activeModalWidget():
            return
            
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._refresh_task = asyncio.ensure_future(self._async_refresh())
            else:
                loop.run_until_complete(self._async_refresh())
        except RuntimeError:
            # No event loop
            pass
    
    async def _async_refresh(self):
        """Async refresh of task data."""
        from src.core.tasks.models import TaskRecord
        
        try:
            # Get active tasks (pending + running)
            active_tasks = await TaskRecord.find({
                "status": {"$in": ["pending", "running"]}
            })
            self._update_active_table(active_tasks)
            
            # Get recent completed (last 10)
            completed_tasks = await TaskRecord.find({
                "status": {"$in": ["completed", "failed"]}
            })
            # Sort by created_at descending and take last 10
            completed_tasks.sort(key=lambda t: t.created_at or 0, reverse=True)
            self._update_recent_table(completed_tasks[:10])
            
            # Update stats
            self._update_stats(active_tasks)
            
        except Exception as e:
            logger.debug(f"Background panel refresh failed: {e}")
    
    def _update_active_table(self, tasks: List):
        """Update active tasks table."""
        self.active_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            # Task name
            name_item = QTableWidgetItem(task.name or "Unknown")
            self.active_table.setItem(row, 0, name_item)
            
            # Handler
            handler_item = QTableWidgetItem(task.handler_name or "")
            self.active_table.setItem(row, 1, handler_item)
            
            # Status with color
            status = task.status or "pending"
            status_item = QTableWidgetItem(status.upper())
            if status == "running":
                status_item.setForeground(QColor("#4CAF50"))  # Green
            elif status == "pending":
                status_item.setForeground(QColor("#FFC107"))  # Yellow
            self.active_table.setItem(row, 2, status_item)
            
            # Progress bar in cell
            progress = task.progress or 0
            progress_item = QTableWidgetItem(f"{progress}%")
            self.active_table.setItem(row, 3, progress_item)
    
    def _update_recent_table(self, tasks: List):
        """Update recent completed table."""
        self.recent_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            # Name
            name_item = QTableWidgetItem(task.name or "Unknown")
            self.recent_table.setItem(row, 0, name_item)
            
            # Handler
            handler_item = QTableWidgetItem(task.handler_name or "")
            self.recent_table.setItem(row, 1, handler_item)
            
            # Result with color based on status
            status = task.status or ""
            if status == "completed":
                result_text = task.result or "OK"
                result_item = QTableWidgetItem("✓ " + result_text[:30])
                result_item.setForeground(QColor("#4CAF50"))
            else:
                error_text = task.error or "Failed"
                result_item = QTableWidgetItem("✗ " + error_text[:30])
                result_item.setForeground(QColor("#F44336"))
            
            self.recent_table.setItem(row, 2, result_item)
    
    def _update_stats(self, active_tasks: List):
        """Update queue stats labels."""
        # Worker count from config
        worker_count = 3
        if hasattr(self.locator, 'config'):
            try:
                worker_count = self.locator.config.data.general.task_workers
            except Exception:
                pass
        
        self.workers_label.setText(f"Workers: {worker_count}")
        
        # Phase counts from pipeline if available
        phase2_count = 0
        phase3_count = 0
        
        if self._pipeline:
            phase2_count = len(getattr(self._pipeline, '_phase2_pending', set()))
            phase3_count = len(getattr(self._pipeline, '_phase3_pending', set()))
        
        self.phase2_label.setText(f"Phase 2: {phase2_count}")
        self.phase3_label.setText(f"Phase 3: {phase3_count}")
