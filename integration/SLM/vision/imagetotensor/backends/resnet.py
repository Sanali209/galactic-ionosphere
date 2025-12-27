
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image
import torch_directml

class CNN_Encoder_ResNet50(CNN_Encoder):
    format = "ResNet50"
    vector_size = 2048

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.device = torch_directml.device()
        self.model = self.model.to(self.device)
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
        inputh_bach = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            futures = self.model(inputh_bach)
            futures = futures.squeeze().cpu()
            encoding = futures.numpy()
            encoding=encoding.flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_ResNet50.format] = CNN_Encoder_ResNet50


module_load()
