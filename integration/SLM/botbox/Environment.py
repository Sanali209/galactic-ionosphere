import random
import time
import uuid

import cv2
import keyboard as keyboardh
import pyautogui
import pygame
import sys
import numpy as np
import pynput
import easyocr
from PIL import Image
from pygame.sprite import Sprite

from typing import TypeVar

# Define a generic type variable T, which is a subtype of the Controller base class
T = TypeVar('T')


class EnvironmentEntity:
    def __init__(self):
        self.name = 'environment_entity'
        self.enabled = True
        self.controllers = []
        self.parentEntity = None
        self.childEntities = []
        self.last_update_time = time.time()
        self.update_interval = 1 / 60

    def SetUpdateRate(self, fps):
        self.update_interval = 1 / fps

    @property
    def env(self) -> 'Environment':
        return self.parentEntity.env

    def on_add_parent_change(self, old_parent, new_parent):
        pass

    def AddChild(self, entity):
        old_parent = entity.parentEntity
        if old_parent is not None:
            old_parent.childEntities.remove(entity)
        self.childEntities.append(entity)
        entity.parentEntity = self
        entity.on_add_parent_change(old_parent, self)

    def RemoveChild(self, entity):
        self.childEntities.remove(entity)
        entity.parentEntity = None

    def GetChild(self, name):
        for entity in self.childEntities:
            if entity.name == name:
                return entity
        return None

    def GetChildByType(self, _type: type[T]) -> T:
        for entity in self.childEntities:
            if isinstance(entity, _type):
                return entity

    def AddController(self, controller):
        old_parent = controller.parentEntity
        self.controllers.append(controller)
        controller.parentEntity = self
        controller.on_add_parent_change(old_parent, self)

    def RemoveController(self, controller):
        self.controllers.remove(controller)
        controller.parentEntity = None

    def GetController(self, name):
        for controller in self.controllers:
            if controller.name == name:
                return controller
        return None

    def GetControllerByType(self, _type: type[T]) -> T:
        for controller in self.controllers:
            if isinstance(controller, _type):
                return controller

    def clear(self):
        for controller in self.controllers:
            controller.clear()
        for entity in self.childEntities:
            entity.clear()

    def update(self):
        if self.enabled:
            for controller in self.controllers:
                controller.update()
            for entity in self.childEntities:
                entity.update()

    def render(self):
        for controller in self.controllers:
            controller.render()
        for entity in self.childEntities:
            entity.render()

    def step(self):
        for controller in self.controllers:
            controller.step()
        for entity in self.childEntities:
            entity.step()


class EntityController:
    def __init__(self):
        self.name = 'entity_controller'
        self.parentEntity: EnvironmentEntity = None
        self.last_update_time = time.time()
        self.update_interval = 1 / 60
        self.is_ai = False
        self.enabled = True

    @property
    def env(self) -> 'Environment':
        return self.parentEntity.env

    def on_add_parent_change(self, old_parent, new_parent):
        pass

    def SetUpdateRate(self, fps):
        self.update_interval = 1 / fps

    def update(self):
        current_time = time.time()
        since_last_update = current_time - self.last_update_time
        if current_time - self.last_update_time > self.update_interval and self.enabled:
            self.last_update_time = current_time
            return True, since_last_update
        return False, since_last_update

    def render(self):
        pass

    def reset(self):
        pass

    def clear(self):
        pass

    def step(self):
        pass


class Transform(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'transform'
        self.position = Vector2d(0, 0)
        self.rotation = 0
        self.scale = Vector2d(1, 1)

    def move(self, x, y):
        self.position = self.position + Vector2d(x, y)

    def rotate(self, angle):
        self.rotation += angle

    def scale(self, x, y):
        self.scale = self.scale * Vector2d(x, y)


class Pawn(EnvironmentEntity):
    def __init__(self, name='pawn'):
        super().__init__()
        self.name = name
        self.transform = Transform()
        self.AddController(self.transform)

    def on_add_parent_change(self, old_parent, new_parent):
        pass

    def render(self):
        pass

    def step(self):
        if not self.enabled:
            return
        for controller in self.controllers:
            if controller.is_ai:
                action = controller.get_action()
                if action is not None:
                    action.do(self)
                reward = 0
                done = False
                if done:
                    controller.set_reward(reward)


class PawnController(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'pawn_controller'
        self.pawn = None

    def reset(self):
        self.pawn = self.parentEntity

    def update(self):
        need_update = super().update()
        if not need_update:
            return

        if self.pawn is None:
            return

class botTtrackEntity:
    def __init__(self):
        self.position = Vector2d(0, 0)
        self.class_name = "None"
        self.inst_name = uuid.uuid4()

class Vector2d:
    def __init__(self, x, y):
        self.values = [x, y]

    @property
    def x(self):
        return self.values[0]

    @property
    def y(self):
        return self.values[1]

    @x.setter
    def x(self, value):
        self.values[0] = value

    @y.setter
    def y(self, value):
        self.values[1] = value

    def to_tuple(self):
        return (self.x,self.y)

    def __eq__(self, other):
        return self.values == other.values

    # override the + operator
    def __add__(self, other):
        return Vector2d(self.values[0] + other.values[0], self.values[1] + other.values[1])

    # override the - operator
    def __sub__(self, other):
        return Vector2d(self.values[0] - other.values[0], self.values[1] - other.values[1])

    def __mul__(self, other):
        if type(other) == Vector2d:
            return Vector2d(self.values[0] * other.values[0], self.values[1] * other.values[1])
        if type(other) == int or type(other) == float:
            return Vector2d(self.values[0] * other, self.values[1] * other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Vector2d({self.x},{self.y})"

    def length(self):
        return (self.values[0] ** 2 + self.values[1] ** 2) ** 0.5

    def scale(self,size:float ):
        return Vector2d(self.values[0] * size, self.values[1] * size)


    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector2d(0, 0)
        return Vector2d(self.values[0] / length, self.values[1] / length)


class Path:
    def __init__(self):
        self.path = []
        self.closed = False

    def AddPoint(self, point: Vector2d):
        self.path.append(point)

    def getNextPoint(self, current_point: Vector2d):
        if len(self.path) == 0:
            return None
        if current_point in self.path:
            index = self.path.index(current_point)
            if index + 1 < len(self.path):
                return self.path[index + 1]
            elif self.closed:
                return self.path[0]
        return None

    def getNearestPoint(self, point: Vector2d):
        if len(self.path) == 0:
            return None
        min_dist = 999999999
        nearest_point = None
        for p in self.path:
            dist = (point - p).length()
            if dist < min_dist:
                min_dist = dist
                nearest_point = p
        return nearest_point


class Camera(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'camera'
        self.position = [0, 0]
        self.rotation = 0
        self.scale = [1, 1]
        self.view_size = [800, 600]

    def move(self, x, y):
        self.camera_position = (self.camera_position[0] + x, self.camera_position[1] + y)

    def rotate(self, angle):
        self.camera_rotation += angle

    def scale(self, x, y):
        self.camera_scale = (self.camera_scale[0] * x, self.camera_scale[1] * y)


class RenderD(EntityController):
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.name = 'render_controller'

    def render_frame_start(self):
        pass

    def cls(self):
        pass

    def draw_text(self, text, pos, color=(255, 255, 255)):
        pass

    def draw_rect(self, pos, size, color=(255, 255, 255)):
        pass

    def draw_image(self, image, pos):
        pass

    def draw_sprite_group(self, sprite_group):
        pass

    def draw_sprite(self, sprite: Sprite):
        pass

    def display_update(self):
        pass


class RenderQtCv(EntityController):
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.name = 'render_controller'
        self.cv_window_name = 'render'
        self.render_size = (800, 600)
        self.cv_image = np.zeros((600, 800, 3), np.uint8)
        self.b_tr = None

    def render_frame_start(self):
        self.cls()

    def cls(self):
        pass

    #cv2.rectangle(self.cv_image, (0, 0), (self.render_size[0], self.render_size[1]), (0, 0, 0), -1)

    def draw_text(self, text, pos, color=(255, 255, 255)):
        try:
            cv2.putText(self.cv_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except:
            pass

    def draw_rect(self, pos, size, color=(255, 255, 255)):
        cv2.rectangle(self.cv_image, pos, (pos[0] + size[0], pos[1] + size[1]), color, 1)

    def draw_image(self, image, pos):
        pilimage = Image.fromarray(image)
        paste_image = Image.fromarray(self.cv_image)
        paste_image.paste(pilimage, pos)
        cv_image = np.array(paste_image)
        self.cv_image = cv_image

    def draw_sprite_group(self, sprite_group):
        pass

    def draw_sprite(self, sprite: Sprite):
        pass

    def display_update(self):
        self.b_tr.set_render_buffer(self.cv_image)
        time.sleep(0.01)


class Render(EntityController):
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.name = 'render_controller'
        pygame.init()
        self.game_screen_size = 800, 600
        self.board_screen = pygame.display.set_mode(self.game_screen_size)

        self.font_size = 16
        self.font = pygame.font.Font(None, self.font_size)

    def render_frame_start(self):
        self.cls()

    def cls(self):
        self.board_screen.fill((0, 0, 0))

    def draw_text(self, text, pos, color=(255, 255, 255)):
        text = self.font.render(text, True, color)
        self.board_screen.blit(text, pos)

    def draw_rect(self, pos, size, color=(255, 255, 255)):
        pygame.draw.rect(self.board_screen, color, (pos[0], pos[1], size[0], size[1]), 1)

    def draw_image(self, image, pos):
        image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        self.board_screen.blit(image, pos)

    def draw_sprite_group(self, sprite_group):
        sprite_group.draw(self.board_screen)

    def draw_sprite(self, sprite: Sprite):
        image = sprite.image
        pos = sprite.rect.topleft
        self.board_screen.blit(image, pos)

    def display_update(self):
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


class OCRModule:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.image = None
        self.result = None

    def set_image(self, image):
        self.image = image

    def read(self):
        self.result = self.reader.readtext(self.image)


class MouseObserver(EntityController):

    def __init__(self):
        super().__init__()
        self.name = 'MouseObserver'
        self.SetUpdateRate(30)
        self.mouse_button_state = set()
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_button_state.add(button)
        else:
            try:
                self.mouse_button_state.remove(button)
            except KeyError:
                pass

    def is_mouse_button_pressed(self, button_name):
        if button_name == 'left':
            left_state = pynput.mouse.Button.left in self.mouse_button_state
            return left_state

    def update(self):
        need_update, last_time = super().update()
        if not need_update:
            return
        data_board = self.parentEntity.GetController('data_board')
        mouse_pos = pyautogui.position()
        mouse_pos = (mouse_pos[0], mouse_pos[1])
        data_board.data["mouse_pos"] = mouse_pos
        data_board.data["mouse_button_state"] = self.mouse_button_state


class ScreenCapturer(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'screen_capturer'
        self.grab_region = (0, 0, 1920, 1080)
        self.update_interval = 1 / 60
        self.ext_update = False
        self.grab_buffer = {}
        self.filters = []


    def set_grab_region(self, region):
        self.grab_region = region

    def add_filter(self,filter_function):
        self.filters.append(filter_function)

    def filter_grab_buffer(self):
        image = self.grab_buffer.get('full_screen_cv')
        if image is None:
            return

        #image = image[200:600, 200:600]

        # convert to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive Equalization
        #image = exposure.equalize_adapthist(image, clip_limit=0.03)

        # convert float 64 to uint8
        #image = cv2.convertScaleAbs(image, alpha=255)

        self.grab_buffer['full_grey_cv'] = image
        for filter_f in self.filters:
            filter_f(self)

    def Grab(self):
        image = pyautogui.screenshot(region=self.grab_region)
        image = np.array(image)
        self.grab_buffer['full_screen_pil'] = image
        self.grab_buffer['full_screen_cv'] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.filter_grab_buffer()

    def update(self):
        need_update, last_time = super().update()
        if not need_update:
            return
        self.Grab()

    def render(self):
        return
        image = self.grab_buffer.get('grey')
        if image is None or image.sum() == 0:
            return
        self.env.renderer.draw_image(image, (0, 0))


class GameScreenRegionFinder(EntityController):
    def __init__(self, name='game_screen_region_finder', template_image=None):
        super().__init__()
        self.grouped_rectangles = []
        self.name = name
        self.template_image = template_image
        self.threshold = 0.8
        self.update_interval = 5 / 60

    def update(self):
        need_update, last_time = super().update()
        if not need_update:
            return
        screen_image = self.parentEntity.GetController('screen_capturer').grabbed_image
        if screen_image is not None:
            cv_screen_image = cv2.cvtColor(np.array(screen_image), cv2.COLOR_RGB2BGR)
            cv_template_image = cv2.cvtColor(np.array(self.template_image), cv2.COLOR_RGB2BGR)
            result = cv2.matchTemplate(cv_screen_image, cv_template_image, cv2.TM_CCOEFF_NORMED)
            location = np.where(result >= self.threshold)
            rectangles = []
            for pt in zip(*location[::-1]):
                rectangles.append((pt[0], pt[1], self.template_image.size[0], self.template_image.size[1]))
            self.grouped_rectangles = rectangles
            data_board = self.parentEntity.GetController('data_board')
            if len(self.grouped_rectangles) > 0:
                self.grouped_rectangles = [rectangles[0]]

                data_board.data[self.name] = self.grouped_rectangles
                data_board.data[self.name + '_count'] = len(self.grouped_rectangles)
                center_position = (self.grouped_rectangles[0][0] + self.grouped_rectangles[0][2] // 2,
                                   self.grouped_rectangles[0][1] + self.grouped_rectangles[0][3] // 2)
                data_board.data[self.name + '_center'] = center_position
            else:
                data_board.data[self.name] = []
                data_board.data[self.name + '_count'] = 0
                data_board.data[self.name + '_center'] = (0, 0)

    def render(self):
        render = self.parentEntity.env.renderer
        for (x, y, w, h) in self.grouped_rectangles:
            render.draw_rect((x, y), (w, h), (0, 255, 0))


class LinearGaugeObserver(EntityController):
    def __init__(self, name='linear_gauge_observer', all_gauge_template=None, observe_indicator_template=None):
        super().__init__()
        self.name = name
        self.all_gauge_template = cv2.cvtColor(np.array(all_gauge_template), cv2.COLOR_RGB2BGR)
        self.observe_indicator_template = cv2.cvtColor(np.array(observe_indicator_template), cv2.COLOR_RGB2BGR)
        self.threshold = 0.3
        self.update_interval = 5 / 60
        self.center_width_normed = 0.0

    def update(self):
        need_update, last_time = super().update()
        if not need_update:
            return
        screen_image = self.parentEntity.GetController('screen_capturer').grabbed_image
        if screen_image is None:
            return
        cv_screen_image = cv2.cvtColor(np.array(screen_image), cv2.COLOR_RGB2BGR)
        board_result = cv2.matchTemplate(cv_screen_image, self.all_gauge_template, cv2.TM_CCOEFF_NORMED)
        board_location = np.where(board_result >= self.threshold)
        board_rectangle = (0, 0, 0, 0)
        for pt in zip(*board_location[::-1]):
            board_rectangle = (pt[0], pt[1], self.all_gauge_template.shape[1], self.all_gauge_template.shape[0])
            break
        indicator_result = cv2.matchTemplate(cv_screen_image, self.observe_indicator_template, cv2.TM_CCOEFF_NORMED)
        indicator_location = np.where(indicator_result >= self.threshold)
        indicator_rectangle = (0, 0, 0, 0)
        for pt in zip(*indicator_location[::-1]):
            indicator_rectangle = (pt[0], pt[1], self.observe_indicator_template.shape[1],
                                   self.observe_indicator_template.shape[0])
            break
        if board_rectangle[2] != 0 and indicator_rectangle[2] != 0:
            self.center_width_normed = (indicator_rectangle[0] - board_rectangle[0]) / board_rectangle[2]
            data_board = self.parentEntity.GetController('data_board')
            data_board.data[self.name] = self.center_width_normed
        else:
            data_board = self.parentEntity.GetController('data_board')
            data_board.data[self.name] = 0.0


class Environment(EnvironmentEntity):
    def __init__(self):
        super().__init__()
        from SLM.botbox.data_board import DataBoard
        self.name = 'bot_environment'
        self.entity_dict = {}
        self.renderer = RenderQtCv()
        self.AddController(self.renderer)
        self.data_board = DataBoard()
        self.AddController(self.data_board)

    @property
    def env(self):
        return self

    def Start(self):
        # todo: implement asink call
        self.clear()
        while not keyboardh.is_pressed('p'):
            self.update()
            self.step()
            self.render()

    def render(self):
        self.renderer.render_frame_start()
        super().render()
        self.renderer.display_update()

    def close(self):
        sys.exit(0)


class Action:
    def __init__(self, name='action'):
        self.name = name

    def do(self, pawn):
        pass

    def cancel(self):
        pass


class Image_region:
    def __init__(self):
        self.name = 'image_region'
        self.region: tuple[int, int, int, int] = (0, 0, 0, 0)

    def SetRegion(self, region: tuple[int, int, int, int]):
        self.region = region

    @staticmethod
    def Region_fromxyxy(x1, y1, x2, y2):
        return tuple(map(int, (x1, y1, x2 - x1, y2 - y1)))

    def Set_region_relative(self, screen_region, shift, size):
        x1 = screen_region[0] + shift[0]
        y1 = screen_region[1] + shift[1]
        self.region = tuple(map(int, (x1, y1, size[0], size[1])))


class Action_region(Action):
    def __init__(self):
        super().__init__()
        self.name = 'action_region'
        self.region: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.click_type = 'left'
        self.click_per_action = 5

    def SetRegion(self, region: tuple[int, int, int, int]):
        self.region = region

    @staticmethod
    def Region_fromxyxy(x1, y1, x2, y2):
        return tuple(map(int, (x1, y1, x2 - x1, y2 - y1)))

    def Set_region_relative(self, screen_region, shift, size):
        x1 = screen_region[0] + shift[0]
        y1 = screen_region[1] + shift[1]
        self.region = tuple(map(int, (x1, y1, size[0], size[1])))

    def do(self):
        min_x = self.region[0]
        max_x = self.region[0] + self.region[2]
        rand_x = np.random.randint(min_x, max_x)
        rand_y = np.random.randint(self.region[1], self.region[1] + self.region[3])
        pyautogui.click(rand_x, rand_y, clicks=self.click_per_action, interval=0.05)


class Discreet_action_space:
    def __init__(self, actions):
        self.name = 'discreet_action_space'
        self.actions = actions
        self.num_actions = len(actions)

    def sample(self):
        import random
        return random.choice(self.actions)

    def get_action_by_name(self, name):
        for action in self.actions:
            if action.name == name:
                return action
        return None


class BotAgent(PawnController):
    def __init__(self):
        super().__init__()
        self.is_ai = True
        self.name = 'bot_agent'
        self.action_space = Discreet_action_space([Action('nothing')])
        self.pause = True

    def get_action(self):
        if self.pause:
            return self.action_space.get_action_by_name('nothing')
        return self.action_space.sample()

    def set_reward(self, reward):
        pass


class KeyDownAction(Action):

    def __init__(self, name, keys=None):
        super().__init__(name)
        self.keys = keys
        if keys is None:
            self.keys = []

    def do(self, pawn):
        rand_time = 0
        # delay
        pyautogui.PAUSE = rand_time
        for key in self.keys:
            pyautogui.keyDown(key)

    def cancel(self):
        for key in self.keys:
            pyautogui.keyUp(key)

    def __eq__(self, other):
        if isinstance(other, KeyDownAction):
            return self.keys == other.keys
        return False


class KeyUpAction(Action):
    def __init__(self, name, keys=None):
        super().__init__(name)
        if keys is None:
            self.keys = []

    def do(self, pawn):
        rand_time = random.randint(20, 50) / 1000
        # delay
        pyautogui.PAUSE = rand_time
        for key in self.keys:
            pyautogui.keyUp(key)
