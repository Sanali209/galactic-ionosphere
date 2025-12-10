from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime

class MsgLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"

class MsgTopic(str, Enum):
    SYSTEM = "SYSTEM"
    SCANNER = "SCANNER"
    AI = "AI_ENGINE"
    DB = "DATABASE"

@dataclass
class SystemMessage:
    level: MsgLevel
    topic: str
    body: str                 # Human-readable text
    payload: Optional[Any] = None # Data (dict, object, exception)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
