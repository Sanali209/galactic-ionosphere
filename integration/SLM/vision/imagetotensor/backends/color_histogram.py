
import cv2

from SLM.vision.cnnhelper import CNN_Helper
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image


class CNN_Encoder_ColorHistogram(CNN_Encoder):
    format = "ColorHistogram"

    def __init__(self):
        super().__init__()
        self.name = "CNN_Encoder_ColorHistogram"
        self.model_name = "ColorHistogram"

    def GetEncoding_by_path(self, image_path):
        return None

    def GetEncoding_from_PilImage(self, image: Image):
        opencvimage = CNN_Helper.PilImage_to_OpenCV_image(image)
        hist = cv2.calcHist([opencvimage], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_ColorHistogram.format] = CNN_Encoder_ColorHistogram


module_load()
