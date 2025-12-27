import timm
import torch
from PIL import Image
from torchvision import transforms
from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor


class CNN_Encoder_DINO(CNN_Encoder):
    format = "DINO"
    vector_size = 384

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.model = self.model.to(self.device)
        self.model.eval()

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path,copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(image)
            output = self.model.forward_head(features, pre_logits=True)
            features = output.squeeze().cpu()
            encoding = features.numpy()
            encoding = encoding.flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_DINO.format] = CNN_Encoder_DINO


module_load()

if __name__ == "__main__":
    image_path = r"E:\rawimagedb\repository\safe repo\asorted images\3\37019507112_f2d61af76a_b.jpg"
    print(ImageToCNNTensor.get_tensor_from_path(image_path, backend="DINO"))
    print(len(ImageToCNNTensor.get_tensor_from_path(image_path, backend="DINO")))
