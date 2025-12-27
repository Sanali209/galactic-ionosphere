import pyglet


class PyGletRender:
    def __init__(self):
        self.game_screen_size = (800, 600)
        self.window = pyglet.window.Window(width=self.game_screen_size[0], height=self.game_screen_size[1])
        
