import torch
import timm
import argparse
from torchvision import transforms
from PIL import Image

from SLM.appGlue.core import Allocator, GlueApp
from SLM.files_db.components.fs_tag import TagRecord
from dataset import ImageDataset
config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"

QtApp = GlueApp()
DB_URL = "mongodb://localhost:27017/"
DB_NAME = "files_db"
MODEL_PATH = "vit_multilabel.pth"
IMAGE_SIZE = 224
list_tags = TagRecord.find({"autotag": True})
SELECTED_TAGS = [tag.fullName for tag in list_tags]
dataset = ImageDataset(DB_URL, DB_NAME, SELECTED_TAGS, image_size=IMAGE_SIZE, use_augmentation=False)
tag_to_idx = dataset.tag_to_idx
idx_to_tag = {v: k for k, v in tag_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=dataset.num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict(image_path, threshold=0.5, top_n=5):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output).cpu().numpy()[0]

    # Выбираем топ-N вероятных тегов
    sorted_tags = sorted(zip(probabilities, idx_to_tag.values()), reverse=True)
    top_tags = [(tag, round(prob, 2)) for prob, tag in sorted_tags[:top_n] if prob > threshold]

    return top_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for tag probability")
    args = parser.parse_args()

    tags = predict(args.image_path, threshold=args.threshold)
    print("Predicted tags:", tags)
