from SLM.vision.LLMBackend import LLMBackend
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class AlexNetLLMBackend(LLMBackend):
    format = "AlexNet"

    def load(self):
        super().load()
        self.model = models.alexnet(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.features.children()))
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

class CNN_Encoder_ImageSearch_AlexNet(CNN_Encoder):
    format = "AlexNet"

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.features.children()))
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def GetEncoding_by_path(self, image_path):
        return None

    def GetEncoding_from_PilImage(self, image: Image):
        image = image.convert('RGB')
        image_tensor = self.preprocess(image)
        inputh_bach = image_tensor.unsqueeze(0)
        with torch.no_grad():
            futures = self.model(inputh_bach)
            futures = futures.squeeze()
            encoding = futures.numpy()
            encoding = encoding.flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_ImageSearch_AlexNet.format] = CNN_Encoder_ImageSearch_AlexNet


module_load()
