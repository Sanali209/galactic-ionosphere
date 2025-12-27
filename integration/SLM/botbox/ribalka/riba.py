import random

import pyautogui
from PIL import Image

from SLM.botbox.Environment import Bot_environment, Pawn, PawnController, MouseObserver, BotAgent, ScreenCapturer, \
    GameScreenRegionFinder, LinearGaugeObserver, Discreet_action_space, Action


class MouseDownAction(Action):
    def __init__(self, name):
        super().__init__(name)

    def do(self, pawn):
        rand_time = random.randint(20, 50) / 1000
        # delay
        pyautogui.PAUSE = rand_time
        pyautogui.mouseDown(button='left')


class MouseUpAction(Action):
    def __init__(self, name):
        super().__init__(name)

    def do(self, pawn):
        rand_time = random.randint(20, 50) / 1000
        # delay
        pyautogui.PAUSE = rand_time
        # Simulate a left mouse button release
        pyautogui.mouseUp(button='left')


class BotRibak(BotAgent):
    def __init__(self):
        super().__init__()
        self.name = 'ribac'
        self.action_space = Discreet_action_space(
            [Action("nothing"), MouseDownAction('mouseDown'), MouseUpAction('mouseUp')])

    def get_action(self):
        env = self.parentEntity.env
        databoard = env.data_board
        ribalka_indicator = databoard.data['ribalka_indicator_center'][0]

        rand_shift = random.randint(0, 18)
        if ribalka_indicator > 50 - rand_shift and ribalka_indicator < 180 + rand_shift:
            databoard.data['ribac_action'] = 'mouseDown'
            return self.action_space.actions[1]
        else:
            get_prew_action = databoard.data.get('ribac_action', 'nothing')
            if get_prew_action == 'mouseDown':
                databoard.data['ribac_action'] = 'nothing'
                return self.action_space.actions[2]


if __name__ == '__main__':
    env = Bot_environment()
    screen_grabber = ScreenCapturer()
    screen_grabber.grab_region = (800, 450, 320, 240)
    env.addController(screen_grabber)
    mouseObserver = MouseObserver()
    env.addController(mouseObserver)
    poplavok_image = Image.open('poplavok.png')
    ribalka_indicator_observer = GameScreenRegionFinder("ribalka_indicator", poplavok_image)
    env.addController(ribalka_indicator_observer)

    ribac_pawn = Pawn('ribac')
    bot = BotRibak()
    ribac_pawn.AddController(bot)
    env.addEntity('ribak', ribac_pawn)
    env.Start()
