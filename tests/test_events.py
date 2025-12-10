import pytest
from src.core.events import ObserverEvent

def test_observer_subscribe_emit():
    event = ObserverEvent("test_evt")
    results = []
    
    def callback(payload):
        results.append(payload)
        
    event.connect(callback)
    event.emit("hello")
    
    assert len(results) == 1
    assert results[0] == "hello"

def test_observer_disconnect():
    event = ObserverEvent("test_evt")
    results = []
    
    def callback():
        results.append(1)
        
    event.connect(callback)
    event.disconnect(callback)
    event.emit()
    
    assert len(results) == 0

def test_observer_error_safety(caplog):
    """Ensure error in one subscriber doesnt block others"""
    event = ObserverEvent("err_evt")
    results = []
    
    def buggy_callback():
        raise ValueError("Bug")
        
    def worker_callback():
        results.append("ok")
        
    event.connect(buggy_callback)
    event.connect(worker_callback)
    
    event.emit()
    
    assert len(results) == 1
    assert results[0] == "ok"
    assert "Bug" in caplog.text
