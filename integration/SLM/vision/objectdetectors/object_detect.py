from PIL import Image

from SLM.appGlue.core import ServiceBackend, BackendProvider


class object_detector_backend(ServiceBackend):
    format = "default"
    version = "1.0.0"

    def detect(self, image: Image):
        return []

    def detect_by_path(self, image_path: str) -> any:
        return []

class ObjectDetectorProvider(BackendProvider):
    name = "ObjectDetectorProvider"

