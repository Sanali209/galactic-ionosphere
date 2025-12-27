from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class CNN_encoder_ModileNetv3_Small(CNN_Encoder):
    format = "ModileNetV3Small"
    vector_size = 576

    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def GetEncoding_from_PilImage(self, image: Image):
        image = image.convert('RGB')
        image_tensor = self.preprocess(image)
        inputh_bach = image_tensor.unsqueeze(0)
        with torch.no_grad():
            futures = self.model(inputh_bach)
            futures = futures.squeeze()
            encoding = futures.numpy()
            encoding=encoding.flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_encoder_ModileNetv3_Small.format] = CNN_encoder_ModileNetv3_Small


module_load()
