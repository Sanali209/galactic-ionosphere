import torch
import torchvision.transforms as transforms
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor
from PIL import Image
import timm
import torch.nn as nn

# --- MobileNet-based Embedding Model ---
class MobileNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(MobileNetEmbeddingNet, self).__init__()
        # Load a pretrained MobileNetV2 model.
        self.backbone = timm.create_model("mobilenetv2_100", pretrained=True)
        # Remove the classification head.
        self.backbone.classifier = nn.Identity()
        # Use the backbone's output features (usually 1280 for MobileNetV2).
        backbone_output_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1280
        # Add a linear layer to project to the desired embedding dimension.
        self.fc = nn.Linear(backbone_output_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        # Normalize embeddings to unit length.
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# --- Custom CNN Encoder using MobileNetEmbeddingNet ---
class CNN_Encoder_mv2_custom(CNN_Encoder):
    format = "CNN_Encoder_mv2_custom"
    vector_size = 512

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the MobileNet-based model.
        self.model = MobileNetEmbeddingNet(embedding_dim=512).to(self.device)
        # Load the pretrained checkpoint for MobileNetEmbeddingNet.
        # Make sure to update the file path to point to your MobileNet checkpoint.
        self.model.load_state_dict(torch.load(r"D:\Sanali209\Python\SLM\vision\imagetotensor\custom_mobile_net\checkpoints\best_model.pth", map_location=self.device))
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
        input_batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_batch)
            embedding = embedding.squeeze().cpu()
            encoding = embedding.numpy().flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_mv2_custom.format] = CNN_Encoder_mv2_custom


module_load()
