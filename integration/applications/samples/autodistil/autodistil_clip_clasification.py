from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our CLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CLIP(
    ontology=CaptionOntology(
        {
            "a person": "person",
            "a two person": "2 person",
            "a three person": "3 person",
            "other": "other",
        }
    )
)

results = base_model.predict(r"E:\rawimagedb\repository\safe repo\asorted images\3\37025612813_a89de98981_b.jpg")

print(results)

base_model.label(r"E:\rawimagedb\repository\safe repo\asorted images\3", extension=".jpg")