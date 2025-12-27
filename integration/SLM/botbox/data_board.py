import cv2
import numpy as np

from SLM.botbox.Environment import EntityController


class DataBoard(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'data_board'
        self.data = {}
        self.SetUpdateRate(5)
        self.draw_image_buf = np.zeros((640, 480, 3), np.uint8)

    def update(self):
        need_update = super().update()
        if not need_update:
            return


    def render(self):
        return
        self.draw_image_buf.fill(0)
        for i, (key, value) in enumerate(self.data.items()):
            cv2.putText(self.draw_image_buf, f"{key}: {value}", (10, i * 16 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('DataBoard', self.draw_image_buf)

