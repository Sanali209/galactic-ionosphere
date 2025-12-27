from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

ontology = CaptionOntology(
    {
        "face": "face",
    }
)

model = GroundingDINO(ontology=ontology)

result = model.predict(r"E:\rawimagedb\repository\safe repo\asorted images\3\37025612813_a89de98981_b.jpg")

plot(
    image=cv2.imread(r"E:\rawimagedb\repository\safe repo\asorted images\3\37025612813_a89de98981_b.jpg"),
    classes=model.ontology.classes(),
    detections=result
)