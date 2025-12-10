import pytest
from src.core.database.models.detection import Detection, DetectedObject
from bson import ObjectId

class TestDetectionModels:
    def test_detection_creation(self):
        """Verify Detection model instantiation and validation."""
        img_id = ObjectId()
        det = Detection(
            parent_image_id=img_id,
            box=[0.1, 0.1, 0.2, 0.2],
            class_label="person",
            confidence=0.95
        )
        assert det.class_label == "person"
        assert det.parent_image_id == img_id
        assert len(det.box) == 4

    def test_detected_object_creation(self):
        """Verify DetectedObject model instantiation."""
        obj = DetectedObject(name="Person A", type="person")
        assert obj.name == "Person A"
        assert obj.type == "person"
