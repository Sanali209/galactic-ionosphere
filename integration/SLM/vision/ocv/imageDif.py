import cv2
import numpy as np
from PIL import Image

from SLM.files_data_cache.pool import PILPool


class OCV_diference_hailaiter:
    def __init__(self, image1_path, image2_path):
        pilimage1 = PILPool.get_pil_image(image1_path)
        pilimage2 = PILPool.get_pil_image(image2_path)
        cvimage1 = cv2.cvtColor(np.array(pilimage1), cv2.COLOR_RGB2BGR)
        cvimage2 = cv2.cvtColor(np.array(pilimage2), cv2.COLOR_RGB2BGR)

        self.image1 = cvimage1
        self.image2 = cvimage2
        self.image1_resized = cv2.resize(self.image1, (720, 720))
        self.image2_resized = cv2.resize(self.image2, (720, 720))
        self.diff = cv2.absdiff(self.image1_resized, self.image2_resized)
        self.gray_diff = cv2.cvtColor(self.diff, cv2.COLOR_BGR2GRAY)
        _,self.thresholded = cv2.threshold(self.gray_diff, 30, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(self.thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def drawImageMaches(self):
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                cv2.rectangle(self.image1_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Differences Highlighted', self.image1_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def DrawImageMachesSideBySide(self):
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                cv2.rectangle(self.image1_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Differences Highlighted', np.hstack([self.image1_resized, self.image2_resized]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

