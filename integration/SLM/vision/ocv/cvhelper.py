import cv2

import numpy as np


class TemplateMatchHelper:

    def __init__(self, imagePath, templatePath):
        self.image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
        self.imagehailaited = self.image
        self.template = cv2.imread(templatePath, cv2.IMREAD_UNCHANGED)
        self.templatesize = self.template.shape

        self.loc = None
        self.rects = None

    def match(self, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
        try:
            result = cv2.matchTemplate(self.image, self.template, method)
            self.loc = np.where(result >= threshold)
        except Exception as e:
            pass

    def CreateRects(self):
        self.rects = []
        try:
            for pt in zip(*self.loc[::-1]):
                self.rects.append((pt[0], pt[1], self.templatesize[1], self.templatesize[0]))
        except Exception as e:
            pass

    def GroupRectangles(self, groupThreshold=1, eps=0.2):
        self.rects, weights = cv2.groupRectangles(self.rects, groupThreshold, eps)

    def drawImageMaches(self):
        for (x, y, w, h) in self.rects:
            cv2.rectangle(self.imagehailaited, (x, y), (x + w, y + h), (0, 255, 0), 2)


    def ShowMachImage(self):
        cv2.imshow('Detected', self.imagehailaited)
        cv2.waitKey(0)


    @staticmethod
    def GetAllMethods(self):
        methods = [func for func in dir(cv2) if func.startswith('TM_')]
        return methods

    @staticmethod
    def IsMach(filepath1, template):
        tmh = TemplateMatchHelper(filepath1, template)
        tmh.match()
        tmh.CreateRects()
        tmh.GroupRectangles()
        return len(tmh.rects) > 0



if __name__ == '__main__':
    tempmach = TemplateMatchHelper(r"D:\Sanali209\Python\SLM\vision\ocv\Screenshot_3.png",
                                   r"D:\Sanali209\Python\SLM\vision\ocv\Screenshot_1.png")

    tempmach.match( 0.8, cv2.TM_CCOEFF_NORMED)
    tempmach.CreateRects()
    tempmach.GroupRectangles()
    tempmach.drawImageMaches()
    tempmach.ShowMachImage()
