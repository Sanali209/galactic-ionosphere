from typing import Dict, Any, Optional
from datetime import datetime
from src.core.database.orm import CollectionRecord, FieldPropInfo

class TaskRecord(CollectionRecord, table="tasks", indexes=["status", "task_type", "created_at"]):
    """
    Persistent record of a background task.
    Status: pending, running, completed, failed
    """
    task_type = FieldPropInfo("task_type", field_type=str)
    status = FieldPropInfo("status", default="pending", field_type=str)
    payload = FieldPropInfo("payload", default={}, field_type=dict)
    
    # Results & Errors
    result = FieldPropInfo("result", default=None, field_type=dict) # Optional output
    error = FieldPropInfo("error", default="", field_type=str)
    
    # Timestamps
    created_at = FieldPropInfo("created_at", default=0.0, field_type=float)
    started_at = FieldPropInfo("started_at", default=0.0, field_type=float)
    completed_at = FieldPropInfo("completed_at", default=0.0, field_type=float)

    async def mark_running(self):
        self.status = "running"
        self.started_at = datetime.now().timestamp()
        await self.save()

    async def mark_completed(self, result: Dict = None):
        self.status = "completed"
        self.result = result or {}
        self.completed_at = datetime.now().timestamp()
        await self.save()

    async def mark_failed(self, error_msg: str):
        self.status = "failed"
        self.error = error_msg
        self.completed_at = datetime.now().timestamp()
        await self.save()
