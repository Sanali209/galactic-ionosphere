from PIL import Image

from SLM.files_data_cache.pool import PILPool


class CNN_Encoder:
    vector_size = 0
    def __init__(self):
        self.name = "CNN_Encoder"

    def GetEncoding_by_path(self, image_path):
        return None

    def GetEncoding_from_PilImage(self, image: Image):
        return None


class ImageToCNNTensor:
    all_backends = {}

    run_backends = {}

    @staticmethod
    def get_tensor_from_path(image_path: str,
                             backend: str = 'ModileNetV3Small') -> any:

        if backend not in ImageToCNNTensor.all_backends:
            return "no backend"
        if backend not in ImageToCNNTensor.run_backends:
            backend_instance = ImageToCNNTensor.all_backends[backend]()
            ImageToCNNTensor.run_backends[backend] = backend_instance
        else:
            backend_instance = ImageToCNNTensor.run_backends[backend]
        try:
            image = PILPool.get_pil_image(image_path)
        except Exception as e:
            return None
        return backend_instance.GetEncoding_from_PilImage(image)

    @staticmethod
    def get_tensor_from_image(image: Image,
                              backend: str = "ModileNetV3Small") -> any:

        if backend not in ImageToCNNTensor.run_backends:
            backend_instance = ImageToCNNTensor.all_backends[backend]()
            ImageToCNNTensor.run_backends[backend] = backend_instance
        else:
            backend_instance = ImageToCNNTensor.run_backends[backend]
        return backend_instance.GetEncoding_from_PilImage(image)

    @staticmethod
    def get_all_backends():
        return ImageToCNNTensor.all_backends.keys()
