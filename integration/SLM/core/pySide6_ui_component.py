"""
PySide6 UI Component for SLM Framework

Integrates PySide6/Qt applications with SLM's synchronous component system.
Pure synchronous implementation using Qt's native threading (QThread).

This is a fallback/simpler alternative to pySide6_ui_async_bridge.py
for users who don't need async Qt operations.
"""

import sys
import threading
from typing import Dict, Any, Optional, List, Callable
from loguru import logger

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QThread, Signal, QObject, QTimer
from PySide6.QtGui import QIcon

from SLM.core.component import Component


class QtSignalBridge(QObject):
    """
    Bridge for Qt signals to communicate with SLM framework.
    Thread-safe communication between Qt and SLM components.
    """

    # Signals for communication
    signal_emitted = Signal(str, dict)  # signal_name, data
    response_received = Signal(str, object)  # callback_id, result

    def __init__(self):
        super().__init__()
        self.callbacks: Dict[str, Callable] = {}
        self.callback_counter = 0

    def register_callback(self, callback: Callable) -> str:
        """Register a callback with unique ID"""
        callback_id = f"cb_{self.callback_counter}"
        self.callback_counter += 1
        self.callbacks[callback_id] = callback
        return callback_id

    def remove_callback(self, callback_id: str):
        """Remove a callback"""
        self.callbacks.pop(callback_id, None)

    def emit_signal(self, signal_name: str, data: Dict[str, Any]):
        """Emit a signal from Qt to SLM framework"""
        self.signal_emitted.emit(signal_name, data)

    def handle_response(self, callback_id: str, result: Any):
        """Handle response from SLM framework"""
        self.response_received.emit(callback_id, result)


class QtApplicationThread(QThread):
    """
    Thread that runs the Qt application event loop.
    Runs Qt in its own thread separate from SLM framework.
    """

    started = Signal()
    error_occurred = Signal(str)

    def __init__(self, args=None):
        super().__init__()
        self.args = [sys.argv[0]] if args is None else args
        self.app: Optional[QApplication] = None
        self.main_window: Optional[QWidget] = None
        self.is_running = False
        self._stop_requested = False

    def set_main_window(self, window: QWidget):
        """Set the main window for the application"""
        self.main_window = window

    def run(self):
        """Run the Qt application in this thread"""
        try:
            self.app = QApplication(self.args)

            # Set application properties
            self.app.setApplicationName("SLM Integrated Qt App")
            self.app.setApplicationVersion("1.0.0")
            self.app.setOrganizationName("SLM Framework")

            if self.main_window:
                self.main_window.show()

            self.is_running = True
            self.started.emit()

            # Start the event loop
            exit_code = self.app.exec()
            logger.info(f"Qt application exited with code: {exit_code}")

        except Exception as e:
            self.error_occurred.emit(f"Qt application error: {e}")
            logger.error(f"Qt application thread error: {e}")

        finally:
            self.is_running = False

    def stop(self):
        """Stop the Qt application"""
        self._stop_requested = True
        if self.app and self.is_running:
            self.app.quit()

    def is_app_running(self) -> bool:
        """Check if the Qt application is running"""
        return self.is_running and self.app is not None


class PySide6UIComponent(Component):
    """
    SLM Component that integrates PySide6/Qt applications.

    Pure synchronous implementation using Qt's native threading.
    Simpler alternative to async bridge for basic Qt applications.

    Features:
    - Sync lifecycle methods (SLM framework compatible)
    - Qt runs in separate thread (QThread)
    - Thread-safe MessageBus integration
    - Window management
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "pyside6_ui")
        self.qt_thread: Optional[QtApplicationThread] = None
        self.signal_bridge: Optional[QtSignalBridge] = None
        self.qt_windows: Dict[str, QWidget] = {}
        self.window_counter = 0

        # Qt application settings
        self.qt_app_args: List[str] = [sys.argv[0]]
        self.qt_app_name = "SLM Qt Application"
        self.qt_organization = "SLM Framework"

        # Thread synchronization
        self._lock = threading.Lock()

    def on_initialize(self):
        """Initialize the PySide6 UI component (sync method)"""
        self.signal_bridge = QtSignalBridge()

        # Connect signals for framework communication
        self.signal_bridge.signal_emitted.connect(self._handle_qt_signal)
        self.signal_bridge.response_received.connect(self._handle_response_signal)

        # Initialize Qt thread
        self.qt_thread = QtApplicationThread(self.qt_app_args)

        # Connect thread signals
        self.qt_thread.started.connect(self._on_qt_started)
        self.qt_thread.error_occurred.connect(self._on_qt_error)

        logger.info("PySide6 UI component initialized")

    def on_start(self):
        """Start the Qt application thread (sync method)"""
        if self.qt_thread and not self.qt_thread.isRunning():
            self.qt_thread.start()

            # Wait for Qt to start (with timeout)
            timeout = 5.0
            import time
            start_time = time.time()
            while not self.qt_thread.is_app_running() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.qt_thread.is_app_running():
                raise RuntimeError("Qt application failed to start within timeout")

            logger.info("PySide6 UI component started")

    def on_stop(self):
        """Stop the Qt application (sync method)"""
        if self.qt_thread:
            self.qt_thread.stop()
            
            # Wait for thread to finish
            timeout = 5.0
            import time
            start_time = time.time()
            while self.qt_thread.isRunning() and (time.time() - start_time) < timeout:
                time.sleep(0.1)

        logger.info("PySide6 UI component stopped")

    def on_shutdown(self):
        """Shutdown the component (sync method)"""
        # Clean up Qt resources
        self.qt_windows.clear()
        logger.info("PySide6 UI component shutdown")

    def on_config_changed(self, key: str, old_value, new_value):
        """Handle configuration changes"""
        if key.startswith('ui.'):
            config_key = key[3:]

            if config_key == 'app_name' and isinstance(new_value, str):
                self.qt_app_name = new_value
                if self.qt_thread and self.qt_thread.app:
                    self.qt_thread.app.setApplicationName(new_value)

            elif config_key == 'organization' and isinstance(new_value, str):
                self.qt_organization = new_value
                if self.qt_thread and self.qt_thread.app:
                    self.qt_thread.app.setOrganizationName(new_value)

    # Public API

    def register_window(self, window: QWidget, window_id: Optional[str] = None) -> str:
        """
        Register a Qt window with the component.

        Args:
            window: The QWidget to register
            window_id: Optional custom ID, auto-generated if not provided

        Returns:
            The window ID assigned to this window
        """
        with self._lock:
            if window_id is None:
                self.window_counter += 1
                window_id = f"window_{self.window_counter}"

            self.qt_windows[window_id] = window

            # Connect window close event
            window.destroyed.connect(lambda: self._on_window_closed(window_id))

            logger.debug(f"Registered window {window_id}: {window.__class__.__name__}")
            return window_id

    def show_window(self, window_id: str):
        """Show a registered window"""
        window = self.qt_windows.get(window_id)
        if window and self.qt_thread and self.qt_thread.is_app_running():
            self._execute_on_qt_thread(window.show)

    def hide_window(self, window_id: str):
        """Hide a registered window"""
        window = self.qt_windows.get(window_id)
        if window and self.qt_thread and self.qt_thread.is_app_running():
            self._execute_on_qt_thread(window.hide)

    def close_window(self, window_id: str):
        """Close a registered window"""
        window = self.qt_windows.get(window_id)
        if window and self.qt_thread and self.qt_thread.is_app_running():
            self._execute_on_qt_thread(window.close)

    def execute_on_ui_thread(self, func: Callable, *args, **kwargs):
        """
        Execute a function on the Qt UI thread.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        if not self.qt_thread or not self.qt_thread.is_app_running():
            logger.warning("Qt application is not running")
            return

        def wrapper():
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing function on Qt thread: {e}")

        self._execute_on_qt_thread(wrapper)

    # Internal methods

    def _execute_on_qt_thread(self, func: Callable):
        """Execute function on Qt thread using QTimer"""
        QTimer.singleShot(0, func)

    def _handle_qt_signal(self, signal_name: str, data: Dict[str, Any]):
        """Handle signals from Qt thread"""
        logger.debug(f"Received Qt signal: {signal_name} with data: {data}")
        
        # Route to message bus if needed
        if self.message_bus:
            self.message_bus.publish(f"ui.{signal_name}", **data)

    def _handle_response_signal(self, callback_id: str, result: Any):
        """Handle response signals"""
        if self.signal_bridge and callback_id in self.signal_bridge.callbacks:
            callback = self.signal_bridge.callbacks[callback_id]
            try:
                callback(result)
            finally:
                self.signal_bridge.remove_callback(callback_id)

    def _on_qt_started(self):
        """Handle Qt application started"""
        logger.info("Qt application started successfully")
        
        # Notify via message bus
        if self.message_bus:
            self.message_bus.publish("ui.qt_started")

    def _on_qt_error(self, error_msg: str):
        """Handle Qt application errors"""
        logger.error(f"Qt application error: {error_msg}")
        
        # Notify via message bus
        if self.message_bus:
            self.message_bus.publish("ui.qt_error", error=error_msg)

    def _on_window_closed(self, window_id: str):
        """Handle window closed events"""
        with self._lock:
            if window_id in self.qt_windows:
                del self.qt_windows[window_id]
                logger.debug(f"Window {window_id} closed")
                
                # Notify via message bus
                if self.message_bus:
                    self.message_bus.publish("ui.window_closed", window_id=window_id)

    # Utility methods

    def get_registered_windows(self) -> List[str]:
        """Get list of registered window IDs"""
        with self._lock:
            return list(self.qt_windows.keys())

    def is_qt_running(self) -> bool:
        """Check if Qt application is running"""
        return self.qt_thread and self.qt_thread.is_app_running()

    def get_window(self, window_id: str) -> Optional[QWidget]:
        """Get a registered window by ID"""
        with self._lock:
            return self.qt_windows.get(window_id)

    def get_qt_app(self) -> Optional[QApplication]:
        """Get the Qt application instance"""
        return self.qt_thread.app if self.qt_thread else None
