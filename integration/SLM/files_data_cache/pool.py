import threading

import PIL
from PIL import Image
from loguru import logger

import time

from SLM.appGlue.timertreaded import TimerBuilder


class PoolItem:
    def __init__(self, item):
        self.item = item
        self.life_start = time.time()


def on_timer_notify(self):
    if PILPool.verbose:
        logger.info("PILPool items: " + str(len(PILPool.pool)))
    for_delete = []
    for key in PILPool.pool:
        if PILPool.pool[key].life_start + PILPool.life_in_seconds < time.time():
            for_delete.append(key)
    for key in for_delete:
        del PILPool.pool[key]


class PILPool:
    pool = {}
    life_in_seconds = 60
    verbose = False
    timer = TimerBuilder().set_interval(30).set_on_timer_notyfy(on_timer_notify).build()



    @staticmethod
    def get_pil_image(path, copy=True) -> 'Image':
        if path is None:
            return Image.new("RGB", (32, 32), (255, 0, 0))
        if path not in PILPool.pool:
            try:
                image = Image.open(path)
                image = image.convert("RGB")
                PILPool.pool[path] = PoolItem(image)
            except Exception as e:
                logger.exception(e)
                PILPool.pool[path] = PoolItem(Image.new("RGB", (32, 32), (255, 0, 0)))
        im: Image = PILPool.pool[path].item
        if copy:
            return im.copy()
        return im

