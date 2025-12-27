from SLM.botbox.Environment import Action


class RPGMouseSteering(Action):
    def __init__(self, name):
        super().__init__(name)
        self.vector_lenth = 100
        self.movement_vector = [0, 0]  #normalized vector
        self.screen_zerro_coord = [950, 406]

    def do(self, pawn):
        if self.movement_vector[0] == 0 and self.movement_vector[1] == 0:
            return
        mouse_vector = [self.movement_vector[0] * self.vector_lenth, self.movement_vector[1] * self.vector_lenth]
        position = (self.screen_zerro_coord[0] + mouse_vector[0], self.screen_zerro_coord[1] + mouse_vector[1])
        print(f"Moving to {position}")
        #pyautogui.mouseDown(position[0], position[1],'right')

    def cancel(self):
        pass
        #pyautogui.mouseUp('right')
