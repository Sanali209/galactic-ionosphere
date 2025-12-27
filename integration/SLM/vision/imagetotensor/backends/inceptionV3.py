import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from SLM.files_data_cache.pool import PILPool
from SLM.vision.imagetotensor.CNN_Encoding import CNN_Encoder, ImageToCNNTensor


class CNN_Encoder_InceptionV3(CNN_Encoder):
    format = "InceptionV3"
    vector_size = 2048  # The desired feature vector size

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the InceptionV3 model with aux_logits=True as expected.
        self.model = models.inception_v3(pretrained=True, aux_logits=True)

        # Replace the final fully-connected layer with an identity mapping.
        # In eval mode, the forward pass returns only the main branch output.
        self.model.fc = torch.nn.Identity()

        self.model.to(self.device)
        self.model.eval()  # Ensure the model is in evaluation mode.

        self.preprocess = transforms.Compose([
            transforms.Resize(299),  # Inception requires 299x299 images.
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def GetEncoding_by_path(self, image_path):
        image = PILPool.get_pil_image(image_path, copy=False)
        return self.GetEncoding_from_PilImage(image)

    def GetEncoding_from_PilImage(self, image: Image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # In evaluation mode, the model returns only the main branch output.
            features = self.model(image_tensor)
            # Squeeze the batch dimension and convert to a 1D numpy array.
            encoding = features.squeeze().cpu().numpy().flatten()
            return encoding


def module_load():
    ImageToCNNTensor.all_backends[CNN_Encoder_InceptionV3.format] = CNN_Encoder_InceptionV3


module_load()
