"""
LLM Job Request/Result Models

Dataclasses for inter-process communication with LLM worker pool.
Uses slots and frozen where possible for performance.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import IntEnum
import uuid
import time


class JobPriority(IntEnum):
    """Job priority levels for queue ordering."""
    HIGH = 0
    NORMAL = 1
    LOW = 2


class JobStatus(IntEnum):
    """Job lifecycle status."""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4


@dataclass(slots=True)
class LLMJobRequest:
    """
    Request for LLM inference job.
    
    Serialized and sent to worker process via multiprocessing.Queue.
    Must be pickle-able (no lambda, no open file handles).
    
    Attributes:
        job_id: Unique identifier for tracking
        task_type: Extractor type (clip, blip, wdtagger, yolo, grounding_dino)
        file_paths: List of absolute file paths to process
        options: Model-specific options (thresholds, phrases, etc.)
        priority: Queue priority (0=HIGH, 1=NORMAL, 2=LOW)
        created_at: Unix timestamp of submission
    """
    task_type: str
    file_paths: List[str]
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    options: Dict[str, Any] = field(default_factory=dict)
    priority: int = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other: "LLMJobRequest") -> bool:
        """Enable priority queue ordering: lower priority value = higher priority."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


@dataclass(slots=True)
class LLMJobResult:
    """
    Result from LLM inference job.
    
    Returned to main process via result queue.
    
    Attributes:
        job_id: Matching request job_id
        success: Whether inference completed successfully
        task_type: Echo of request task_type for routing
        data: Model outputs (embeddings, captions, detections, tags)
        error: Error message if failed
        elapsed_ms: Processing time in milliseconds
    """
    job_id: str
    success: bool
    task_type: str  
    data: Any = None  # Can be dict, list, or any pickle-able result
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    
    @classmethod
    def failure(cls, job_id: str, task_type: str, error: str) -> "LLMJobResult":
        """Create a failure result."""
        return cls(
            job_id=job_id,
            success=False,
            task_type=task_type,
            error=error
        )


# Shutdown sentinel for worker processes
SHUTDOWN_SENTINEL = LLMJobRequest(
    task_type="__SHUTDOWN__",
    file_paths=[],
    job_id="__SHUTDOWN__",
    priority=JobPriority.HIGH
)
