# standard libraries
import torch
from PIL import Image

# optional, for DirectML device selection
import torch_directml

# YOLOv8 + Hugging Face + Supervision
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

from SLM.files_data_cache.pool import PILPool
from SLM.vision.objectdetectors.object_detect import object_detector_backend




class FaceDetectorYolov8HF(object_detector_backend):
    """
    A class to detect faces using a YOLOv8 face model from Hugging Face.
    """

    def __init__(self,
                 repo_id="arnabdhar/YOLOv8-Face-Detection",
                 filename="model.pt",
                 use_directml=False,
                 conf_threshold=0.25):
        """
        Initializes the face detector by downloading the YOLO model from Hugging Face
        and preparing the device (DirectML, CUDA, or CPU).

        Args:
            repo_id (str): The Hugging Face repo ID containing the model.
            filename (str): The name of the .pt file in that repo.
            use_directml (bool): Whether to use Torch DirectML. If False,
                                 automatically uses CUDA if available, else CPU.
            conf_threshold (float): Confidence threshold for YOLO predictions.
        """
        # Decide on device
        if use_directml:
            self.device = torch_directml.device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download model checkpoint from Hugging Face
        print(f"Downloading model from HF: {repo_id}/{filename} ...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        print(f"Model loaded on device: {self.device}, confidence threshold: {self.conf_threshold}")

    def detect_faces(self, image: Image.Image):
        """
        Detect faces in a given PIL image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            list[dict]: A list of detection results. Each dict has:
                - version: str
                - label: str
                - region_format: str (usually "abs_xywh")
                - score: float
                - region: [x, y, w, h]
                - image: cropped PIL image of the face
        """
        # Run YOLOv8 inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device
        )

        # Convert YOLO results to supervision Detections
        detections = Detections.from_ultralytics(results[0])
        if len(detections) == 0:
            return []

        detection_results = []

        # detections.xyxy is [num_detections, 4] => x1, y1, x2, y2
        for xyxy, confidence in zip(detections.xyxy, detections.confidence):
            x1, y1, x2, y2 = xyxy

            # Build a consistent result dict
            res_dict = {
                "version": "1.0.0",
                "label": "face",
                "region_format": "abs_xywh",
                "score": float(confidence),
                "region": [
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1)
                ],
            }

            # Crop face from the original image
            try:
                cropped_face = image.crop((int(x1), int(y1), int(x2), int(y2)))
                res_dict["image"] = cropped_face
            except Exception as e:
                print(f"Error cropping face: {e}")
                continue

            detection_results.append(res_dict)

        return detection_results

    def detect_faces_by_path(self, image_path: str):
        """
        Detect faces by loading an image from a file path.

        Args:
            image_path (str): File path to the image.

        Returns:
            list[dict]: A list of detection results in the same format as detect_faces().
        """
        try:
            # If you're using PILPool or your own loader, replace here
            # image = PILPool.get_pil_image(image_path)
            image = PILPool.get_pil_image(image_path,copy=False)
            print(f"[YOLOv8] Processing image: {image_path}, size: {image.size}")

            return self.detect_faces(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []



# -------------------------
# Usage Example
if __name__ == "__main__":

    # Initialize the face detector (example uses CPU or CUDA automatically)
    face_detector = FaceDetectorYolov8HF()

    # Your local image path
    image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37071967133_389aa53b2d_b.jpg"

    # Detect
    results = face_detector.detect_faces_by_path(image_path)

    # Print or visualize results
    for idx, result in enumerate(results):
        image = result["image"]
        image.show()

        # Show the cropped face image (uncomment if desired)
        # result["image"].show()

    print("Done.")
