"""
UCoreFS Engine Module

Provides the core architecture for running heavy processing in a separate thread
(The Engine) while maintaining a responsive UI (The Client).
"""
from src.core.engine.thread import EngineThread
from src.core.engine.proxy import EngineProxy
