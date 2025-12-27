import numpy as np
from PIL import Image

#todo: put in right place
class CNN_Helper:

    @staticmethod
    def NPArrayToList(array: np.array) -> list:
        return array.tolist()

    @staticmethod
    def ListToNPArray(data_list: list) -> np.array:
        return np.array(data_list)

    @staticmethod
    def DF_image_toPilImage(image):
        image *= 255
        # order of chanels as in opencv reodered to RGB
        #image = image[:, :, ::-1]
        pil_image = Image.fromarray(image.astype(np.uint8))
        del image
        return pil_image

    @staticmethod
    def PilImage_to_DF_image(pil_image):
        image = np.array(pil_image)
        image = image.astype(np.float32) / 255
        return image

    @staticmethod
    def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def findEuclideanDistance(source_representation, test_representation):
        if isinstance(source_representation, list):
            source_representation = np.array(source_representation)

        if isinstance(test_representation, list):
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    @staticmethod
    def l2_normalize(x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))



    @classmethod
    def image_resize(cls, image, target_size):
        """
        simple image resize stretch image to target size
        @param image:
        @param target_size:
        @return:
        """
        if image.size[0] < target_size[0] or image.size[1] < target_size[1]:
            image = image.resize(target_size, Image.BOX)
        if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
            image = image.resize(target_size, Image.ANTIALIAS)
        return image

    @classmethod
    def PilImage_to_OpenCV_image(cls, image):
        import cv2
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
