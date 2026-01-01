"""
UExplorer - Maintenance Settings Page

Displays all maintenance tasks in a table with controls to run them manually.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QHeaderView, QCheckBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from loguru import logger


class MaintenanceSettingsPage(QWidget):
    """Settings page for maintenance task configuration."""
   
    run_task_requested = Signal(str)  # task_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Info
        info_label = QLabel(
            "Maintenance tasks run periodically to ensure data integrity. "
            "You can manually trigger any task below."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #cccccc; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Task table
        tasks_group = QGroupBox("Maintenance Tasks")
        tasks_layout = QVBoxLayout(tasks_group)
        
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(4)
        self.tasks_table.setHorizontalHeaderLabels([
            "Task", "Enabled", "Schedule", "Action"
        ])
        
        # Make table stretch
        header = self.tasks_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        self._populate_tasks()
        
        tasks_layout.addWidget(self.tasks_table)
        layout.addWidget(tasks_group)
        
        layout.addStretch()
    
    def _populate_tasks(self):
        """Populate the tasks table."""
        tasks = [
            ("Count Verification", True, "Every 5 minutes", "background_verification"),
            ("Database Optimization", True, "Daily", "database_optimization"),
            ("Cache Cleanup", True, "Weekly", "cache_cleanup"),
            ("Orphaned File Cleanup", True, "Weekly", "orphaned_cleanup"),
            ("Log Rotation", True, "Daily", "log_rotation"),
            ("Database Cleanup", False, "Monthly", "database_cleanup"),
            ("Diagnose Pipeline State", False, "Manual", "diagnose_pipeline_state"),
            ("Fix File Types", False, "Manual", "fix_file_types"),
            ("Reprocess Incomplete Embeddings", False, "Manual", "reprocess_incomplete_embeddings"),
        ]
        
        self.tasks_table.setRowCount(len(tasks))
        
        for row, (name, enabled, schedule, task_id) in enumerate(tasks):
            # Task name
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tasks_table.setItem(row, 0, name_item)
            
            # Enabled checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(enabled)
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.tasks_table.setCellWidget(row, 1, checkbox_widget)
            
            # Schedule
            schedule_item = QTableWidgetItem(schedule)
            schedule_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tasks_table.setItem(row, 2, schedule_item)
            
            # Run button
            run_btn = QPushButton("Run Now")
            run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #5a7aaa;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #6a8aba;
                }
            """)
            run_btn.clicked.connect(lambda checked, tid=task_id: self.run_task_requested.emit(tid))
            self.tasks_table.setCellWidget(row, 3, run_btn)
