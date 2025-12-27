import json

from PIL import Image
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification

import torch

import torch_directml


class ImageToLabelBackend:
    format = ""
    version = ""

    def __init__(self):
        self.version = "1.0"

    def get_label_from_path(self, image_path, *kwargs) -> any:
        return []


class multiclass_genres_v001(ImageToLabelBackend):
    def __init__(self):
        super().__init__()

        self.pipline_name = "sanali209/imclasif-genres-v001"
        self.pipeline = pipeline("image-classification", model=self.pipline_name, framework="pt")

    def get_label_from_path(self, image_path, *kwargs) -> any:  # return list of tuples (label, confidence)
        response = self.pipeline(image_path)
        return response


class BoundedRegressionModel(torch.nn.Module):
    def __init__(self, base_model, min_val=1.0, max_val=10.0):
        super().__init__()
        self.base = base_model
        self.min_val = min_val
        self.max_val = max_val
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        raw = outputs.logits
        bounded = self.sigmoid(raw) * (self.max_val - self.min_val) + self.min_val
        outputs.logits = bounded
        return outputs


class multiclass_rating_HF(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.model_path = "sanali209/rating1_10"  # локальная папка или HuggingFace ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # для DirectML
        # self.device = torch_directml.device()

        base_model = AutoModelForImageClassification.from_pretrained(self.model_path)
        self.model = BoundedRegressionModel(base_model, min_val=1.0, max_val=10.0).to(self.device)
        self.model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)

    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(pixel_values=inputs["pixel_values"])
                rating = outputs.logits.item()
                return rating
        except Exception as e:
            print(f"Error during inference: {e}")
            return None


class multiclass_NSFW_HF(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = "1.0"
        self.pipline_name = "sanali209/nsfwfilter"
        self.pipeline = pipeline("image-classification", model=self.pipline_name, framework="pt",
                                 device=self.device)
        print("NSFW_HF")

    def get_label_from_path(self, image_path, **kwarg) -> any:
        response = self.pipeline(image_path)
        return response


from openai import OpenAI
import base64
import requests

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

template = "This is a chat between a user and an assistant."


class multiclass_multiclass_NSFW_LLM_LLAVA(ImageToLabelBackend):
    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        image = open(image_path.replace("'", ""), "rb").read()
        base64_image = base64.b64encode(image).decode("utf-8")
        template = "This is a chat between a user and an assistant."
        completion = client.chat.completions.create(
            model="llava-v1.5-7b",
            messages=[
                {
                    "role": "system",
                    "content": template,
                },
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }, {"type": "text", "text":
                            (
                                r"chose one of category: 'not NSFW','NSFW','ero','violent','gore' answer in JSON format:"
                                r"{'category':'chosen category'}"
                                r"{'description':'shot describe the image'}"
                                r"{'probability':'probability of the category'}")
                            },
                    ],
                }
            ],
            max_tokens=1000,
            # stream=True
        )
        response = completion.choices[0].message
        # convert the response to a dictionary
        json_r = json.loads(response.content)

        annotat_label = json_r['category']
        # transform labels
        if annotat_label == "not NSFW":
            annotat_label = "safe"
        if annotat_label == "NSFW":
            annotat_label = "porn"
        if annotat_label == "ero":
            annotat_label = "ero"
        if annotat_label == "violent":
            annotat_label = "explicit"
        if annotat_label == "gore":
            annotat_label = "explicit"

        response = [{"label": annotat_label, "confidence": json_r['probability']}]
        return response


class multiclass_sketch_bf(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.pipline_name = "sanali209/sketch_filter"
        self.pipeline = pipeline("image-classification", model=self.pipline_name, framework="pt")

    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        response = self.pipeline(image_path)
        return response


class multiclass_comix_bf(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.pipline_name = "sanali209/comixBF"
        self.pipeline = pipeline("image-classification", model=self.pipline_name, framework="pt")

    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        response = self.pipeline(image_path)
        return response

    def get_label_from_pil_image(self, pil_image) -> any:
        response = self.pipeline(pil_image)
        return response


class multiclass_image_type_bf(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.pipline_name = "sanali209/imagetypeBF"
        self.pipeline = pipeline("image-classification", model=self.pipline_name, framework="pt")

    def get_label_from_path(self, image_path) -> any:  # return list of tuples (label, confidence)
        response = self.pipeline(image_path)
        return response


class Salesforce_blip_image_captioning_base_tr(ImageToLabelBackend):
    def __init__(self):
        super().__init__()
        self.name = "Salesforce_blip_image_captioning_base"
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    def get_label_from_path(self, image_path) -> any:
        answer = self.pipeline(image_path)
        text = answer[0]['generated_text']
        return text


class ImageToLabel:
    all_backends = {
        "multiclass_genres_v001": multiclass_genres_v001,
        "multiclass_sketch_bf": multiclass_sketch_bf,
        "text_salesforce_blip_image_base": Salesforce_blip_image_captioning_base_tr,
        "multiclass_rating_HF": multiclass_rating_HF,
        "multiclass_NSFW_HF": multiclass_NSFW_HF,
        "multiclass_comix_bf": multiclass_comix_bf,
        "multiclass_image_type_bf": multiclass_image_type_bf,
        "multiclass_multiclass_NSFW_LLM_LLAVA": multiclass_multiclass_NSFW_LLM_LLAVA
    }

    run_backends = {}

    @staticmethod
    def get_label_from_path(image_path: str,
                            backend: str = None, **kwargs) -> any:  # return list of tuples (label, confidence)

        if backend not in ImageToLabel.run_backends:
            if backend not in ImageToLabel.all_backends:
                return "no backend"
            backend_instance = ImageToLabel.all_backends[backend]()
            ImageToLabel.run_backends[backend] = backend_instance
        else:
            backend_instance = ImageToLabel.run_backends[backend]

        return backend_instance.get_label_from_path(image_path, **kwargs)

    @staticmethod
    def get_backend_version(backend: str):
        if backend not in ImageToLabel.run_backends:
            backend_instance = ImageToLabel.all_backends[backend]()
            ImageToLabel.run_backends[backend] = backend_instance
        else:
            backend_instance = ImageToLabel.run_backends[backend]

        return backend_instance.version

    @staticmethod
    def get_all_backends():
        return list(ImageToLabel.all_backends.keys())


class RegressionRatingHF(ImageToLabelBackend):
    """
    A backend for image rating prediction using a Hugging Face regression model.

    This class loads a model fine-tuned for regression (outputting a single value)
    and provides an interface similar to the Hugging Face pipeline.
    """

    def __init__(self, model_id: str, use_directml: bool = False):
        """
        Initializes the regression model backend.

        Args:
            model_id (str): The Hugging Face Hub repository ID or a local path to the model.
            use_directml (bool): Flag to attempt using the DirectML device.
                                 (Requires torch-directml to be installed).
        """
        super().__init__()
        self.model_id = model_id

        # --- Device Selection ---
        self.device = None
        if use_directml:
            try:
                import torch_directml
                self.device = torch_directml.device()
                print("Using DirectML device for inference.")
            except ImportError:
                print("Warning: torch_directml not found. Falling back to CUDA/CPU.")

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model on device: {self.device}")

        # --- Model Loading ---
        # We load the model manually instead of using a pipeline because this is a
        # custom regression task, not standard classification.
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            print(f"Successfully loaded model '{self.model_id}'.")
        except Exception as e:
            print(f"Fatal: Failed to load model from '{self.model_id}'.")
            raise e

    def get_label_from_path(self, image_path: str) -> list:
        """
        Analyzes an image and returns its predicted rating.

        Args:
            image_path (str): The file path to the image.

        Returns:
            list: A list containing a single dictionary, formatted to be
                  pipeline-like: [{'label': 'rating', 'score': 8.53}]
                  Returns an empty list if an error occurs.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return []
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []

        # Perform inference
        with torch.no_grad():
            # Preprocess the image and move tensors to the correct device
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

            # Get model output
            outputs = self.model(**inputs)

            # The raw output is our predicted rating. Use .item() to extract the float.
            predicted_rating = outputs.logits.item()

        # Format the output to be consistent with the pipeline's structure
        response = [{'label': 'rating', 'score': predicted_rating}]
        return response


if __name__ == '__main__':
    print(ImageToLabel.get_label_from_path(r"C:\Users\Sanali\PycharmProjects\SLM\test\test.jpg",
                                           "multiclass_genres_v001"))
