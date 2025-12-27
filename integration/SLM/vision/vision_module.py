from SLM import Allocator
from SLM.appGlue.core import Module


class VisionModule(Module):
    def __init__(self):
        super().__init__("VisionModule")

    def init(self):
        from SLM.vision.objectdetectors.backends.autodistil.face.ground_dino_face import object_detector_groundDino
        from SLM.vision.objectdetectors.object_detect import ObjectDetectorProvider
        from SLM.vision.objectdetectors.backends.torch.mtcnn import FaceDetectorMTCNN
        from SLM.vision.objectdetectors.backends.torch.yolov8 import FaceDetectorYolov8HF

        object_detector_provider = ObjectDetectorProvider()
        Allocator.res.register(object_detector_provider)
        object_detector_provider.register_backend(object_detector_groundDino())
        object_detector_provider.register_backend(FaceDetectorMTCNN())
        object_detector_provider.register_backend(FaceDetectorYolov8HF())