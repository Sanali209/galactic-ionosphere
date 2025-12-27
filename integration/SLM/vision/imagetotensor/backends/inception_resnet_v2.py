import torch
import timm
import torchvision.transforms as transforms

from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image
import torch_directml

class CNN_Encoder_InceptionResNetV2(CNN_Encoder):
    format = "InceptionResNetV2"
    vector_size = 1536  # Размер выходного вектора для Inception-ResNet-V2

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model('inception_resnet_v2', pretrained=True)
        # Оставляем только часть модели до последнего слоя
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception-ResNet-V2 requires 299x299 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path,copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        image = image.convert('RGB')
        image_tensor = self.preprocess(image)
        input_batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_batch)
            features = features.squeeze().cpu()
            encoding = features.numpy()
            encoding = encoding.flatten()
        return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_InceptionResNetV2.format] = CNN_Encoder_InceptionResNetV2


module_load()
