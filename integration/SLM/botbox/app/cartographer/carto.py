import cv2
import numpy as np
import pyautogui
import pynput
from PIL import Image

from SLM.botbox.Environment import Environment, Pawn, PawnController, MouseObserver, BotAgent, ScreenCapturer, \
    GameScreenRegionFinder, LinearGaugeObserver, Discreet_action_space, Action, EnvironmentEntity, \
      EntityController
import keyboard as keyboardhandler


class AlbionZoneMap2(EnvironmentEntity):
    def __init__(self):
        super().__init__()
        self.name = 'AlbionZoneMap'
        self.map_size = (2000, 2000)
        self.tile_size = 100
        self.obstacle_color = (148, 99, 74)
        self.map_data = np.zeros((self.map_size[1], self.map_size[0], 3), np.uint8)

    def load_from_minimap(self,minimap_image_path):
        minimap_image = cv2.imread(minimap_image_path)
        minimap_image = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2RGB)
        self.map_data = cv2.resize(minimap_image, (self.map_size[0], self.map_size[1]), interpolation=cv2.INTER_AREA)

    def render(self):
        super().render()
        cv2.imshow('Map', self.map_data)

class AlbionZoneMap(EnvironmentEntity):
    def __init__(self):
        super().__init__()
        self.name = 'AlbionZoneMap'
        self.map_size = (1000, 1000)
        self.currentMapImage = Image.open('init.png')
        self.currentMapImage = cv2.cvtColor(np.array(self.currentMapImage), cv2.COLOR_RGB2BGR)
        keypoints1, descriptors1 = find_keypoints_and_descriptors(self.currentMapImage)
        self.keypoints = keypoints1
        self.descriptors = descriptors1
        self.screen_pos = [0,0]
        self.player_position = [0,0]
        self.movement_vector = [0,0]


    def render(self):
        super().render()
        cv2.imshow('Map', self.currentMapImage)


class MapGrabber(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MapGrabber'
        self.update_interval = 100
        self.paused = False
        self.keyborde_listner = pynput.keyboard.Listener(on_press=self.on_press)
        self.keyborde_listner.start()
        self.prew_frame = None
        self.time_from_last_update = 0
        self.last_update_time_stump = 0

    def on_press(self, key):
        try:
            if key.char == 'n':
                self.paused = not self.paused
        except AttributeError:
            pass

    def grab_screen_and_merge(self):
        screen_image = self.parentEntity.env.getController('screen_capturer').cv_grabbed_image
        zoneMap: AlbionZoneMap = self.parentEntity

        if self.prew_frame is None:
            self.prew_frame = screen_image
            return

        # Преобразование изображений в оттенки серого
        prev_gray = cv2.cvtColor(self.prew_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # Вычислить оптический поток
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Вычислить среднее смещение

        # Вычислить среднее смещение
        flow_mean = np.mean(flow, axis=(0, 1))
        shift_x, shift_y = flow_mean

        # Переместить кадр
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped_frame = cv2.warpAffine(screen_image, M,
                                      (zoneMap.currentMapImage.shape[1], zoneMap.currentMapImage.shape[0]))




        alpha = 0.7
        # Наложить кадр
        zoneMap.currentMapImage = cv2.addWeighted(zoneMap.currentMapImage, alpha, warped_frame, 1-alpha, 0)

        cv2.imshow('Map', zoneMap.currentMapImage)

    def update(self):
        needUpdate = super().update()
        if not needUpdate or self.paused:
            return
        self.grab_screen_and_merge()


if __name__ == '__main__':
    env = Environment()
    screen_grabber = ScreenCapturer()
    screen_grabber.grab_region = (300, 200, 640, 480)
    env.AddController(screen_grabber)
    mouseObserver = MouseObserver()
    env.AddController(mouseObserver)
    zoneMap = AlbionZoneMap()
    env.AddChild(zoneMap)
    #mapGrabber = MapGrabber()
    #zoneMap.AddController(mapGrabber)

    env.Start()
