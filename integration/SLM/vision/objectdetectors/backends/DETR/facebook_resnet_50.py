from transformers import DetrImageProcessor, DetrForObjectDetection

from PIL import Image

import torch

# TODO: test and integrate this model
class ObjectDetectorRect_DETR_model:
    name = "facebook_resnet_50"
    version = "1.0.0"
    """
    use huggingface transformers DETR model for object detection
    """
    pipline_name = "facebook/detr-resnet-50"
    detector_name = "object_detector_rectangle/detr_facebook_detr_resnet_50"

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained(self.pipline_name)
        self.model = DetrForObjectDetection.from_pretrained(self.pipline_name)
        self.score_threshold = 0.5
        self.box_format = "abs_xyxy"

    def predict(self, image: Image):
        """
        predict object from image
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        result = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                              threshold=self.score_threshold)[0]
        pred_result = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_value = self.model.config.id2label[label.item()]
            pred_result.append(
                {"prediction_type": "object_detection",
                 "name": self.pipline_name,
                 "version": "1.0.0",
                 "label": label_value,
                 "score": score.item(),
                 "region": box,
                 "box_format": "abs_xyxy",

                 }
            )

        return pred_result


# todo realise my custom face_head_char model detector
class ObjectDetectorRect_DETR_model_face_head_char:
    """
    use custom  huggingface transformers DETR model for object detection
    """
    pipline_name = "sanali209/DT_face_head_char"


