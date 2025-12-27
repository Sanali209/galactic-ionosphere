import os
import random
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

import pymongo
import timm  # Make sure to install timm via pip
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.files_data_cache.pool import PILPool
from SLM.files_data_cache.thumbnail import ImageThumbCache

config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_services()


# --- Custom Dataset ---
class RelationDataset(Dataset):
    def __init__(self, mongodb_uri="mongodb://localhost:27017", db_name="your_db_name", transform=None):
        super().__init__()
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.transform = transform

        # Placeholders to be filled in prepare_items
        self.image_paths = {}
        self.positive_pairs = []
        self.anchor_to_negatives = {}
        self.all_image_ids = []

        # Prepare dataset (load from MongoDB)
        self.prepare_items()

    def prepare_items(self):
        """Loads data from MongoDB and prepares positive/negative pairs."""
        client = pymongo.MongoClient(self.mongodb_uri)  # Connect to MongoDB
        db = client[self.db_name]

        # Load image records
        collection_records = db["collection_records"]
        for record in tqdm(collection_records.find({"item_type": "FileRecord"}), desc="Loading images"):
            full_path = os.path.join(record["local_path"], record["name"])
            self.image_paths[record["_id"]] = full_path

        # Define relation types
        self.positive_types = {"similar", "near_dub", "similar_style", "some_person", "some_image_set", "other"}
        self.negative_type = "wrong"

        # Load relation records
        relation_records = list(db["relation_records"].find({"type": "similar_search"}))
        for rec in tqdm(relation_records, desc="Loading relations"):
            sub_type = rec.get("sub_type", "").lower()
            from_id, to_id = rec.get("from_id"), rec.get("to_id")

            if sub_type in self.positive_types and from_id in self.image_paths and to_id in self.image_paths:
                self.positive_pairs.append((from_id, to_id))
            elif sub_type == self.negative_type and from_id in self.image_paths and to_id in self.image_paths:
                self.anchor_to_negatives.setdefault(from_id, []).append(to_id)

        # Store all image IDs for random negatives
        self.all_image_ids = list(self.image_paths.keys())
        print(f"Found {len(self.positive_pairs)} positive pairs.")
        client.close()  # Close connection after data is loaded

    def __len__(self):
        return len(self.positive_pairs)

    def _load_image(self, path):
        path = ImageThumbCache.instance().get_thumb(path, "medium")
        image = PILPool.get_pil_image(path)
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        anchor_id, positive_id = self.positive_pairs[idx]
        anchor_path, positive_path = self.image_paths[anchor_id], self.image_paths[positive_id]

        # Select a negative sample
        negative_candidates = self.anchor_to_negatives.get(anchor_id, [])
        negative_id = random.choice(negative_candidates) if negative_candidates else random.choice(self.all_image_ids)
        if negative_id in negative_candidates:
            negative_candidates.remove(negative_id)
        while negative_id == anchor_id or negative_id == positive_id:
            negative_id = random.choice(self.all_image_ids)

        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(self.image_paths[negative_id])

        return {"anchor": anchor_img, "positive": positive_img, "negative": negative_img}


# --- MobileNet-based Embedding Model ---
class MobileNetEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(MobileNetEmbeddingNet, self).__init__()
        # Load a pretrained MobileNetV2 model.
        self.backbone = timm.create_model("mobilenetv2_100", pretrained=True)
        # Remove the classification head by replacing it with an identity module.
        self.backbone.classifier = nn.Identity()
        # Determine the output dimension (typically 1280 for MobileNetV2).
        backbone_output_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1280

        # Add a linear layer to map features to the desired embedding dimension.
        self.fc = nn.Linear(backbone_output_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        # Normalize embeddings to unit length.
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# --- Training Loop with Checkpoint Resume ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Define transforms (adjust as needed)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and split into train and validation.
    full_dataset = RelationDataset(mongodb_uri="mongodb://localhost:27017",
                                   db_name="files_db",
                                   transform=transform)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer.
    model = MobileNetEmbeddingNet(embedding_dim=512).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0

    # Create a directory to save checkpoints.
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check for an existing checkpoint to resume training.
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        train_batches = len(train_loader)
        pbar = tqdm(total=train_batches, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch", leave=False)
        for batch in train_loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = criterion(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({"Loss": loss.item()})
        pbar.close()

        avg_train_loss = epoch_loss / train_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", unit="batch", leave=False)
        with torch.no_grad():
            for batch in val_loader:
                anchor = batch["anchor"].to(device)
                positive = batch["positive"].to(device)
                negative = batch["negative"].to(device)

                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)

                loss = criterion(anchor_embed, positive_embed, negative_embed)
                val_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
        pbar.close()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Save full checkpoint including training state.
        checkpoint_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
        torch.save(checkpoint_dict, latest_ckpt_path)
        print(f"Saved latest checkpoint: {latest_ckpt_path}")

        # Save a separate checkpoint for this epoch.
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch + 1}: {epoch_checkpoint_path}")

        # Save the best model based on validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train_model()
