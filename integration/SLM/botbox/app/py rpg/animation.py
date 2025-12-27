
from  helper import get_frame

class Animation:
    def __init__(self, sheet, frame_rects, frame_duration):
        self.frames = [get_frame(sheet, *rect) for rect in frame_rects]
        self.frame_duration = frame_duration
        self.current_frame = 0
        self.time_accumulator = 0

    def update(self, dt):
        self.time_accumulator += dt
        if self.time_accumulator >= self.frame_duration:
            self.time_accumulator -= self.frame_duration
            self.current_frame = (self.current_frame + 1) % len(self.frames)

    def get_image(self):
        return self.frames[self.current_frame]

