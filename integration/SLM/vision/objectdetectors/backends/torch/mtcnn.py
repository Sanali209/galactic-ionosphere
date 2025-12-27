import torch
from PIL import Image
import torch_directml
from facenet_pytorch import MTCNN

from SLM.vision.objectdetectors.object_detect import object_detector_backend
from SLM.files_data_cache.pool import PILPool


class FaceDetectorMTCNN(object_detector_backend):
    """
    A class for face detection using MTCNN and Torch with optional DirectML acceleration.

    Attributes:
        device (torch.device): The device used for computations (DirectML, CUDA, or CPU).
        detector (MTCNN): The face detection model.
    """

    def __init__(self, use_directml=False):
        """
        Initializes the FaceDetectorTorch class.

        Args:
            use_directml (bool): Whether to use Torch DirectML for acceleration. Defaults to True.
        """
        self.device = torch_directml.device() if use_directml else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(device=self.device)
        print(f"Using device: {self.device}")  # Log the device for debugging

    def detect_faces(self, image: Image):
        """
        Detects faces in a given image.

        Args:
            image (PIL.Image): The input image in which faces are to be detected.
            image_path (str): The file path of the input image (used for cropping).

        Returns:
            list[dict]: A list of detected faces with metadata including coordinates, score, and cropped face image.
        """
        image = image.convert('RGB')

        # Ensure the image tensor is on the correct device
        try:
            boxes, probs = self.detector.detect(image)
        except Exception as e:
            print(f"Detection error: {e}")
            return []

        if boxes is None:
            return []

        detection_results = []
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = box
            res_dict = {"version": "1.0.0", "label": "face", "region_format": "abs_xywh", 'score': float(prob),
                        'region': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]}

            image_size_array = [x1, y1, x2, y2]
            image_size_array = [int(coord) for coord in image_size_array]
            try:
                face_image = image.copy().crop(image_size_array)
                res_dict['image'] = face_image
            except Exception as e:
                print(f"Error cropping face image: {e}")
                continue

            detection_results.append(res_dict)

        return detection_results

    def detect_faces_by_path(self, image_path: str):
        """
        Detects faces in an image specified by its file path.

        Args:
            image_path (str): The file path of the input image.

        Returns:
            list[dict]: A list of detected faces with metadata.
        """
        try:
            image = PILPool.get_pil_image(image_path, copy=False)
            print(f"Processing image: {image_path}, size: {image.size}")  # Log image size
            return self.detect_faces(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []


