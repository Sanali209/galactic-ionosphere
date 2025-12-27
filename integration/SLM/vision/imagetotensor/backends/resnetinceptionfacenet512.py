import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image
import torch_directml




class CNN_Encoder_FaceNet(CNN_Encoder):
    vector_size = 512
    format = "CNN_Encoder_FaceNet"

    def __init__(self,use_directml=False):
        super().__init__()
        if use_directml:
            self.device = torch_directml.device()
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Изменение размера изображения
            transforms.ToTensor(),  # Конвертация в тензор
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
        ])

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path,copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predict = self.model(image_tensor)
            futures = predict.squeeze().cpu()
            futures = futures.numpy()
            encoding = futures.flatten()

        # convert shape from (1, 512) to (512,)
        return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_FaceNet.format] = CNN_Encoder_FaceNet


module_load()
