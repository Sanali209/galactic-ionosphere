import unittest

import cv2
import numpy as np
from autodistill.utils import plot
from supervision import Detections

from SLM.appGlue.DesignPaterns.backend import BackendRepository
from SLM.vision.objectdetectors.object_detect import object_detector_backend


class TestSLMFSDB(unittest.TestCase):

    def test_detect_face(self):
        path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37025612813_a89de98981_b.jpg"
        backends = BackendRepository.get_compatible("object_detector/face/")
        for backend in backends:
            backend: object_detector_backend
            name = BackendRepository.get_name(backend)
            result = backend.detect_by_path(path)

            xyxyd = []
            confidenced = []
            labeld = []
            for res in result:
                labeld.append(0)
                confidenced.append(res["score"])

                xyxyd.append([res["region"][0], res["region"][1],
                              res["region"][0] + res["region"][2], res["region"][1] + res["region"][3]])

            detection = Detections(xyxy=np.array(xyxyd), confidence=np.array(confidenced), class_id=np.array(labeld))
            print(name)
            plot(image=cv2.imread(path),
                 classes=["face"],
                 detections=detection
                 )

    def test_detect_face_backend(self):
        path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37025612813_a89de98981_b.jpg"
        backend: object_detector_backend = BackendRepository.get("object_detector/face/autodistil_groundDINO")

        backend: object_detector_backend

        result = backend.detect_by_path(path)

        xyxyd = []
        confidenced = []
        labeld = []
        for res in result:
            labeld.append(0)
            confidenced.append(res["score"])

            xyxyd.append([res["region"][0], res["region"][1],
                              res["region"][0] + res["region"][2], res["region"][1] + res["region"][3]])

        detection = Detections(xyxy=np.array(xyxyd), confidence=np.array(confidenced), class_id=np.array(labeld))
        plot(image=cv2.imread(path),
                 classes=["face"],
                 detections=detection
                 )