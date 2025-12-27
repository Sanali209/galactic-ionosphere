import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pyautogui
import pynput
from PIL import Image
from PySide6.QtCore import QThread, Signal, Qt, Slot

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QPushButton, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableWidgetItem

from SLM.botbox.Environment import EntityController, Pawn, EnvironmentEntity, Environment, ScreenCapturer, BotAgent, \
    Discreet_action_space, Action
from SLM.botbox.behTree import Sequence, Blackboard, ActionNode
from SLM.groupcontext import group

from helper import cv2pixmap, detect_shift




class YoloV8Detector(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'YoloV8Detector'
        self.update_interval = 1/10
        self.paused = False


class BufferSaver(EntityController):
    def __init__(self):
        super().__init__()
        self.save_dir = 'screenshots'
        self.name = 'BufferSaver'
        self.update_interval = 1/2
        self.enabled = False
        self.buffer = None
        self.buffer_name = 'grey'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def update(self):
        need_update,since_last_update = super().update()
        if not need_update:
            return
        screen_image = self.parentEntity.env.GetControllerByType(ScreenCapturer).grab_buffer.get('full_screen_pil')
        if screen_image is None:
            return
        #convert np array to pil image
        screen_image = Image.fromarray(screen_image)

        # save image to disck
        screen_image.save(f"{self.save_dir}/{time.time()}.png")


class BotEnabler(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'BotEnabler'
        self.update_interval = 1
        self.enabled = False

    def update(self):
        needUpdate = super().update()
        if not needUpdate:
            return

        screen_image = self.parentEntity.env.GetControllerByType(ScreenCapturer).grab_buffer.get('full_screen_pil')
        # get value of pixel 101,36 if [239,213,184]
        bot = self.env.GetChild('p_pwn')
        if (screen_image is None) or (bot is None):
            return
        screen_image = np.array(screen_image)
        if screen_image[36, 101][0] == 239 and screen_image[36, 101][1] == 213 and screen_image[36, 101][2] == 184:
            bot.enabled = True
            self.env.data_board.data['bot_enabled'] = True
        else:
            bot.enabled = False
            self.env.data_board.data['bot_enabled'] = False


class MiniMapAlb(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MiniMapAlb'
        self.mini_map_rect = [1620, 810, 1900, 1040]
        self.map_data = None
        self.player_position = [0, 0]
        self.update_interval = 1/4

    def init(self):
        self.env.data_board.data['mini_map_rect'] = self.mini_map_rect

    def render(self):
        image = self.map_data
        if image is None:
            return
        # downscale to 800*600
        #image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
        self.env.renderer.draw_image(image, (0, 0))

    def update(self):
        needUpdate = super().update()
        if not needUpdate:
            return
        self.grab_mini_map()

    def grab_mini_map(self):
        self.mini_map_rect = self.env.data_board.data.get('mini_map_rect')
        screen_image = self.parentEntity.env.GetController('screen_capturer').grab_buffer['full_screen_pil']
        if screen_image is None:
            return
        # get region
        self.map_data = screen_image[self.mini_map_rect[1]:self.mini_map_rect[3],
                        self.mini_map_rect[0]:self.mini_map_rect[2]]

        target_rgb = [97, 179, 255]

        matches = np.where(np.all(self.map_data == target_rgb, axis=-1))
        if len(matches[0]) == 0:
            self.env.data_board.data['player_position'] = [0, 0]
            return
        # todo improve by find center of mass
        mean_x = int(np.mean(matches[0]))
        mean_y = int(np.mean(matches[1]))

        self.player_position = [mean_x, mean_y]
        self.env.data_board.data['player_position'] = self.player_position


class BotWanderer(BotAgent):
    def __init__(self):
        super().__init__()
        self.name = 'vanderer'
        self.action_space = Discreet_action_space(
            [Action("nothing")])
        self.sequence = Sequence()

        self.sequence.add_child(ActionNode(move_to_target_m))
        self.sequence.add_child(ActionNode(select_new_position))
        self.blackboard = Blackboard()
        self.blackboard.set_value('target_position', (10, 0))

    def update(self):
        super().update()
        zoneMap = self.env.GetController('MiniMapAlb')
        self.blackboard.set_value('current_position', zoneMap.player_position)

    def get_action(self):
        prew_action = self.blackboard.get_value('action')

        self.sequence.run(self.blackboard)
        act = self.blackboard.get_value('action')
        if prew_action is not None and prew_action != act:
            try:
                prew_action.cancel()
            except:
                print("Can't cancel action")
        if act is None:
            return self.action_space.actions[0]
        else:
            pass
        return act




class botThread(QThread):
    render_buffer = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.env = Environment()
        enabler = BotEnabler()
        self.env.AddController(enabler)
        self.env.renderer.b_tr = self
        sc_graber = ScreenCapturer()
        self.env.AddController(sc_graber)
        buffer_saver = BufferSaver()
        self.env.AddController(buffer_saver)
        MiniMap = MiniMapAlb()
        self.env.AddController(MiniMap)
        MiniMap.init()
        #zoneMap = ZoneMap2()
        #self.env.AddChild(zoneMap)
        player_pawn = Pawn("p_pwn")
        player_pawn.enabled = False
        bo = BotWanderer()
        player_pawn.AddController(bo)
        self.env.AddChild(player_pawn)

    def set_render_buffer(self, buffer):
        self.render_buffer.emit(buffer)

    def run(self):
        self.env.Start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boThread = botThread()
        self.boThread.render_buffer.connect(self.update_image)
        self.setWindowTitle("OpenCV with Qt - Tune Canny Edges")
        self.setGeometry(100, 100, 800, 600)

        container = QWidget()
        self.setCentralWidget(container)
        with group():
            layout = QHBoxLayout()
            container.setLayout(layout)
            self.label = QLabel(self)
            self.label.setAlignment(Qt.AlignCenter)
            with group():
                left_layout = QVBoxLayout()
                layout.addLayout(left_layout)
                self.enableButton = QPushButton('Enable')
                self.enableButton.clicked.connect(self.boThread.start)
                self.start_grabB = QPushButton('Start Grab')
                self.start_grabB.clicked.connect(self.start_grab)
                left_layout.addWidget(self.label)
                left_layout.addWidget(self.enableButton)
                left_layout.addWidget(self.start_grabB)
            # Set up the table
            self.table_widget = QTableWidget()
            # on property change
            self.table_widget.setRowCount(5)
            self.table_widget.setColumnCount(2)
            self.table_widget.setHorizontalHeaderLabels(['Parameter', 'Value'])
            layout.addWidget(self.table_widget)



    def start_grab(self):
        self.boThread.env.GetControllerByType(BufferSaver).enabled = True

    def update_table(self, data):
        self.table_widget.setRowCount(len(data))
        for i, (key, value) in enumerate(data.items()):
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            self.table_widget.setItem(i, 0, key_item)
            self.table_widget.setItem(i, 1, value_item)

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        # if cv_image is filed with zeros, do nothing

        # dawn_scale to 800*600
        #cv_img = cv2.resize(cv_img, (800, 600), interpolation=cv2.INTER_AREA)

        q_img = cv2pixmap(cv_img)
        self.label.setPixmap(q_img)
        data_board = self.boThread.env.data_board
        self.update_table(data_board.data)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
