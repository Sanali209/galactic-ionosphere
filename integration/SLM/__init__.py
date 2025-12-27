"""
SLM Framework - Simplified Python Application Framework

The SLM Framework provides both a simple decorator-based API for easy usage
and a powerful underlying system with dependency injection, message bus,
async support, MongoDB ODM, and UI integration.

Quick Start (Simple API):
    import SLM as slm
    
    @slm.component
    class MyService:
        def start(self):
            print("Service started!")
    
    if __name__ == "__main__":
        slm.run()

Advanced Usage:
    from SLM.core.app import App
    from SLM.core.component import Component
    # ... use full framework features
"""




# Legacy compatibility - keep existing imports working
from SLM.appGlue.DAL.DAL import DataConverterFactory
from SLM.appGlue.core import Allocator
from SLM.appGlue.progress_visualize import ProgressManager
from SLM.appGlue.timertreaded import TimerManager

# Initialize legacy components
Allocator.res.register(DataConverterFactory())
Allocator.res.register(TimerManager())
Allocator.res.register(ProgressManager())

# Framework metadata
__version__ = "2.0.0"
__author__ = "SLM Framework Team"
__description__ = "Simplified Python Application Framework with async, DI, and UI support"

# Export list for * imports
__all__ = [

    # Legacy
    'DataConverterFactory', 'Allocator', 'ProgressManager', 'TimerManager'
]
