import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration
from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor


# TODO: Test large BLIP model if necessary
class CNN_Encoder_BLIP(CNN_Encoder):
    format = "BLIP"
    vector_size = 768

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.generation_model =BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        self.model = self.model.to(self.device)
        self.generation_model = self.generation_model.to(self.device)
        self.model.eval()
        self.generation_model.eval()

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path, copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            vision_output = self.model.vision_model(**inputs)
            features = vision_output.pooler_output.cpu().numpy()
            return features.flatten()

    def get_encoding_for_text(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_output = self.model.text_model(**inputs)
            features = text_output.pooler_output.cpu().numpy()
            return features.flatten()

    def best_texts_mach(self, image: Image.Image, texts_t: list[str]):
        """
        Find the best matching text for the image.
        Args:
            image (Image.Image): Input image.
            texts_t (list[str]): List of text strings to compare.
        Returns:
            tuple: Best matching text and normalized similarity scores for all texts.
        """
        image_encoding = torch.tensor(self.GetEncoding_from_PilImage(image))
        distances = []

        best_text = None
        best_similarity = -1

        for text in texts_t:
            text_encoding = torch.tensor(self.get_encoding_for_text(text))
            similarity = torch.nn.functional.cosine_similarity(image_encoding, text_encoding, dim=0).item()
            distances.append(similarity)

            if similarity > best_similarity:
                best_similarity = similarity
                best_text = text

        # Normalize distances to a range of [0, 1]
        max_sim, min_sim = max(distances), min(distances)
        normalized_distances = [(sim - min_sim) / (max_sim - min_sim) for sim in distances]

        return best_text, normalized_distances

    def generate_image_description(self, image: Image.Image,
                                   request="Generate detail description of image.",max_lenth = 250) -> str:
        """
        Generate a textual description for the given image.
        Args:
            image (Image.Image): Input image.
        Returns:
            str: Generated description of the image.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.generation_model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_BLIP.format] = CNN_Encoder_BLIP


module_load()

if __name__ == "__main__":
    image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"

    # Example usage
    encoder = CNN_Encoder_BLIP()

    # Best match example
    best_match, similarities = encoder.best_texts_mach(
        PILPool.get_pil_image(image_path),
        ["woman", "woman on the horse", "man on the horse", "horse", "cat"]
    )
    print(f"Best match: {best_match}")
    print(f"Similarities: {similarities}")

    # Generate image description example
    description = encoder.generate_image_description(PILPool.get_pil_image(image_path))
    print(f"Generated description: {description}")
