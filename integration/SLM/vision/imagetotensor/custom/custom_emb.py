import torch
import torchvision.transforms as transforms
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image
# --- ViT-based Embedding Model (Same as in train.py) ---
import torch.nn as nn
import timm


class ViTEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(ViTEmbeddingNet, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.backbone.reset_classifier(0)
        backbone_output_dim = self.backbone.num_features
        self.fc = nn.Linear(backbone_output_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class CNN_Encoder_custom(CNN_Encoder):
    format = "custom"
    vector_size = 1024

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTEmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(r"D:\Sanali209\Python\SLM\vision\imagetotensor\custom\checkpoints\best_model.pth", map_location=self.device))
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def GetEncoding_by_path(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        image = image.convert('RGB')
        image_tensor = self.preprocess(image)
        inputh_bach = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            futures = self.model(inputh_bach)
            futures = futures.squeeze().cpu()
            encoding = futures.numpy()
            encoding = encoding.flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_custom.format] = CNN_Encoder_custom


module_load()
