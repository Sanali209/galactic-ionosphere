from typing import Any
from .schema import SystemMessage, MsgLevel

class MessageBuilder:
    def info(self, topic: str, body: str) -> SystemMessage:
        return SystemMessage(MsgLevel.INFO, topic, body)

    def error(self, topic: str, body: str, exc: Exception = None) -> SystemMessage:
        return SystemMessage(MsgLevel.ERROR, topic, body, payload={"exception": exc})

    def success(self, topic: str, body: str, data: Any = None) -> SystemMessage:
        return SystemMessage(MsgLevel.SUCCESS, topic, body, payload=data)
