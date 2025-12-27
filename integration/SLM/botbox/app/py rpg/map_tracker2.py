import math
import os
import random
import sys
import time
# from collections import namedtuple # No longer needed for these
from typing import List, Optional, Tuple, Dict  # For type hinting

from kornia.feature import DISKFeatures  # Retain for type hinting kornia's output
from loguru import logger

# Настройка логирования
logger.add(
    "map_tracker.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB"
)

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal, Qt, Slot
from PySide6.QtGui import QCloseEvent

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QPushButton, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableWidgetItem, QSlider

# Assuming these are custom/local modules and are correct
from SLM.botbox.Environment import EntityController, Pawn, Environment, ScreenCapturer
from SLM.groupcontext import group
# from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget # Not used in provided snippet
from SLM.pySide6Ext.widgets.object_editor import PySide6ObjectEditor, ObjectEditorView

from helper import cv2pixmap  # , detect_shift # detect_shift not used in provided snippet


class PlayerTracker(EntityController):
    def __init__(self):
        super().__init__()
        self.slam: Optional[SLAMSystem2DImpl] = None  # Initialize later
        # Frame size from ScreenCapturer.grab_region = (0,100,1700,600) -> W=1700, H=500.
        self.slam_frame_width = 1700
        self.slam_frame_height = 500
        try:
            self.slam = SLAMSystem2DImpl(frame_width=self.slam_frame_width,
                                         frame_height=self.slam_frame_height,
                                         feature_config=None)
            logger.info("PlayerTracker: SLAM system initialized.")
        except Exception as e:
            logger.error(f"PlayerTracker: Failed to initialize SLAM system: {e}", exc_info=True)
            self.slam = None  # Ensure SLAM is None if init fails.

    def update(self):
        if not super().update(): return
        if not self.slam:
            self.env.data_board.data["slam_status"] = "SLAM Not Initialized"
            return

        image = self.env.GetControllerByType(ScreenCapturer).grab_buffer.get("map_tracing_image")
        if image is None:
            logger.trace("PlayerTracker: No map_tracing_image in buffer.")
            return

        try:
            success = self.slam.process_frame(image)
            # Update data board regardless of success to show current status
            self.env.data_board.data["slam_pose_x"] = f"{self.slam.current_pose.x:.2f}"
            self.env.data_board.data["slam_pose_y"] = f"{self.slam.current_pose.y:.2f}"

            if success:
                status = "Tracking" if self.slam.is_initialized else "Initialized"
                self.env.data_board.data["slam_status"] = status
            else:  # process_frame returned False
                status = "Initializing" if not self.slam.is_initialized else "Lost (Process Frame Failed)"
                self.env.data_board.data["slam_status"] = status
                logger.debug(f"PlayerTracker: SLAM process_frame failed. Status: {status}")

        except Exception as e:
            logger.error(f"Exception in PlayerTracker SLAM processing: {e}", exc_info=True)
            self.env.data_board.data["slam_status"] = f"Error: Processing Frame"


class GameWorldMap(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'GameWorldMapVisualizer'  # Renamed for clarity

    def add_marker(self):
        logger.info("Add marker button clicked (GameWorldMap).")


class Visualizer(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'SLAMVisualizer'

    def render(self):
        image_to_draw_on = self.env.GetControllerByType(ScreenCapturer).grab_buffer.get("map_tracing_image")

        resized_image = cv2.resize(image_to_draw_on, (800, 600), interpolation=cv2.INTER_AREA)
        # The renderer is expected to call self.b_tr.set_render_buffer(resized_image)
        self.env.renderer.draw_image(resized_image, (0, 0))


class botThread(QThread):
    render_buffer = Signal(np.ndarray)  # For MainWindow to connect to

    def __init__(self):
        super().__init__()
        self.env = Environment()
        # This assignment allows the renderer (part of env) to signal back to this thread,
        # which then signals to the MainWindow.
        self.env.renderer.b_tr = self

        sc_graber = ScreenCapturer()
        sc_graber.grab_region = (0, 100, 1700, 600)  # W=1700, H=500
        sc_graber.add_filter(filter_for_tracking)
        self.env.AddController(sc_graber)

        self.env.AddController(filters_setings())
        self.env.AddController(GameWorldMap())
        self.env.AddController(PlayerTracker())  # SLAM system is inside this
        self.env.AddController(Visualizer())  # Draws SLAM state

        player_pawn = Pawn("p_pwn")
        player_pawn.enabled = True
        self.env.AddChild(player_pawn)

    def set_render_buffer(self, buffer: np.ndarray):  # Called by Renderer via self.env.renderer.b_tr
        self.render_buffer.emit(buffer)

    def run(self):
        logger.info("Bot thread started.")
        try:
            self.env.Start()  # This is the main blocking loop of the environment
        except Exception as e:
            logger.critical(f"Critical unhandled exception in botThread env.Start(): {e}", exc_info=True)
        finally:
            logger.info("Bot thread run method finished.")
            # self.env.Stop() # Ensure env is stopped if loop exits for any reason - env.Start() should handle its cleanup


class filters_setings(EntityController):
    gaus_kernel_size = 5

    def set_gaus_kernel_size(self, size: int):
        corrected_size = max(1, size if size % 2 != 0 else size + 1)
        self.gaus_kernel_size = corrected_size
        logger.debug(f"Gaussian kernel size set to: {self.gaus_kernel_size}")


def filter_for_tracking(sc_capturer: ScreenCapturer):
    current_frame_cv = sc_capturer.grab_buffer.get("full_screen_cv")
    if current_frame_cv is None:
        sc_capturer.grab_buffer["map_tracing_image"] = None;
        return

    filters_ctrl = sc_capturer.env.GetControllerByType(filters_setings)
    k_size = filters_ctrl.gaus_kernel_size if filters_ctrl else 5

    blurred = cv2.GaussianBlur(current_frame_cv, (k_size, k_size), 0)
    sc_capturer.grab_buffer["map_tracing_image"] = blurred


class filter_settings_editor(ObjectEditorView):
    def __init__(self, obj: filters_setings):
        super().__init__(obj)
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Gaussian Blur Kernel Size (Odd)", self)
        self.layout.addWidget(self.label)

        self.slider_value_label = QLabel(f"Current: {self.object.gaus_kernel_size}")
        self.layout.addWidget(self.slider_value_label)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(1);
        self.slider.setMaximum(25)  # e.g. 1 to 25
        self.slider.setValue(self.object.gaus_kernel_size)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(2);
        self.slider.setSingleStep(1)  # Step by 1, correction in update_value
        self.slider.valueChanged.connect(self.update_value)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

    def update_value(self, value: int):
        self.object.set_gaus_kernel_size(value)  # Method handles making it odd
        self.slider_value_label.setText(f"Current: {self.object.gaus_kernel_size}")
        # If slider value differs from corrected value, update slider to match
        if self.slider.value() != self.object.gaus_kernel_size:
            self.slider.setValue(self.object.gaus_kernel_size)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boThread = botThread()
        self.boThread.render_buffer.connect(self.update_image_and_table)

        self.setWindowTitle("2D SLAM System Visualizer")
        self.setGeometry(50, 50, 1250, 750)  # Adjusted size for table and controls

        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_label = QLabel("Waiting for video stream...");
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600);
        self.image_label.setStyleSheet("border: 1px solid #CCC; background-color: #333;")

        self.enableButton = QPushButton('Start SLAM Thread');
        self.enableButton.clicked.connect(self.start_bot_thread)
        # add_marker_button = QPushButton('Add Marker'); add_marker_button.clicked.connect(self.add_map_marker_gui)

        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.enableButton)
        # left_layout.addWidget(add_marker_button)

        filters_controller = self.boThread.env.GetControllerByType(filters_setings)
        if filters_controller:
            object_editor = PySide6ObjectEditor()
            object_editor.add_view_template(filters_setings, filter_settings_editor)
            object_editor.set_object(filters_controller)
            left_layout.addWidget(object_editor)

        self.table_widget = QTableWidget()
        self.table_widget.setMinimumWidth(400)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.table_widget.setColumnWidth(0, 180);
        self.table_widget.setColumnWidth(1, 200)  # Adjusted widths

        main_layout.addWidget(left_widget);
        main_layout.addWidget(self.table_widget)

    def start_bot_thread(self):
        if not self.boThread.isRunning():
            self.boThread.start();
            self.enableButton.setText("SLAM Thread Running...");
            self.enableButton.setEnabled(False)
        else:
            logger.info("Bot thread already running.")

    # def add_map_marker_gui(self):
    #     gwm_ctrl = self.boThread.env.GetControllerByType(GameWorldMap)
    #     if gwm_ctrl: gwm_ctrl.add_marker()
    #     else: logger.warning("GameWorldMap controller not found for GUI action.")

    def update_table(self, data: dict):
        self.table_widget.setRowCount(len(data))
        for i, (key, value) in enumerate(data.items()):
            self.table_widget.setItem(i, 0, QTableWidgetItem(str(key)))
            self.table_widget.setItem(i, 1, QTableWidgetItem(str(value)))

    @Slot(np.ndarray)
    def update_image_and_table(self, cv_img: np.ndarray):
        if cv_img is not None and cv_img.size > 0:
            self.image_label.setPixmap(cv2pixmap(cv_img))

        data_board = self.boThread.env.data_board
        if data_board: self.update_table(dict(data_board.data))

    def closeEvent(self, event: QCloseEvent):
        logger.info("Main window closing. Attempting to stop bot thread...")
        if self.boThread.isRunning():
            self.boThread.env.Stop()
            if not self.boThread.wait(5000):  # Wait up to 5s
                logger.warning("Bot thread did not finish gracefully in 5s. Terminating...")
                self.boThread.terminate()
                self.boThread.wait()  # Wait for termination
            else:
                logger.info("Bot thread finished gracefully.")
        else:
            logger.info("Bot thread was not running.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
