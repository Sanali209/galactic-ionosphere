import torch
import clip
from PIL import Image

from SLM.files_data_cache.pool import PILPool
from SLM.vision.LLMBackend import LLMBackend
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
import torch_directml

class CLIP_VIT_L_14_LLMBackend(LLMBackend):
    format = "CLIP_VIT_L_14"
    vector_size = 768
    tags = ["image_tensor_encoder","text_tensor_encoder","image_text_tensor_encoder","CLIP"]
    def load(self):
        super().load()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()


class CNN_Encoder_CLIP_DML(CNN_Encoder):
    format = "CLIP_DML"
    vector_size = 768

    def __init__(self, use_directml=True):
        super().__init__()

        # Initialize the DirectML device
        if use_directml:
            self.device = torch_directml.device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path, copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        # Preprocess the image and move it to the DirectML device
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
            features = features.squeeze()
            encoding = features.cpu().numpy()
            encoding = encoding.flatten()
            return encoding

    def get_encoding_for_text(self, text: str):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text)
            features = features.squeeze()
            encoding = features.cpu().numpy()
            encoding = encoding.flatten()
            return encoding

    def best_texts_mach(self, image: Image, texts_t: list[str]):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        texts = clip.tokenize(texts_t).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            # transform to numpy
            similarity = similarity.cpu().numpy().flatten()
            best = texts_t[similarity.argmax()]
            return best,similarity


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_CLIP_DML.format] = CNN_Encoder_CLIP_DML


module_load()

if __name__ == "__main__":
    image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
    #print(ImageToCNNTensor.get_tensor_from_path(image_path, backend="CLIP_DML"))
    res = CNN_Encoder_CLIP_DML().best_texts_mach(PILPool.get_pil_image(image_path), ["woman", "woman on the horse",
                                                                                     "man on the horse", "horse","cat"])
    print(res)
