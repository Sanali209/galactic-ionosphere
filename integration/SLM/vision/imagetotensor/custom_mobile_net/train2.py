import os
import random
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import transforms

import pymongo
import timm  # Make sure to install timm via pip
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.files_data_cache.pool import PILPool
from SLM.files_data_cache.thumbnail import ImageThumbCache

# --- Service Initialization ---
config = Allocator.config
config.fileDataManager.path = r"D:\data\ImageDataManager"
config.mongoConfig.database_name = "files_db"
Allocator.init_services()


# --- Custom Dataset with Augmentation and Persistent Data Split ---
class RelationDataset(Dataset):
    def __init__(
        self,
        mongodb_uri="mongodb://localhost:27017",
        db_name="your_db_name",
        transform=None,
        use_augmentation=False,
        augmentation_transform=None,
        augmentation_probability=1.0
    ):
        """
        Args:
            mongodb_uri (str): URI to connect to MongoDB.
            db_name (str): Name of the database.
            transform (callable, optional): Final transformations to apply after augmentation.
            use_augmentation (bool): Whether to apply augmentation.
            augmentation_transform (callable, optional): Custom augmentation pipeline.
                If None and use_augmentation is True, a default augmentation pipeline is applied.
            augmentation_probability (float): Probability (0 to 1) that augmentation is applied to each image.
        """
        super().__init__()
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.augmentation_probability = augmentation_probability

        # Use default augmentation pipeline if enabled and no custom pipeline is provided.
        if self.use_augmentation and augmentation_transform is None:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        else:
            self.augmentation_transform = augmentation_transform

        # Placeholders for dataset items
        self.image_paths = {}
        self.positive_pairs = []
        self.anchor_to_negatives = {}
        self.all_image_ids = []

        # Load dataset from MongoDB
        self.prepare_items()

    def prepare_items(self):
        """Loads data from MongoDB and prepares positive/negative pairs."""
        client = pymongo.MongoClient(self.mongodb_uri)
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

        # Store all image IDs for negative sampling
        self.all_image_ids = list(self.image_paths.keys())
        print(f"Found {len(self.positive_pairs)} positive pairs.")
        client.close()

    def __len__(self):
        return len(self.positive_pairs)

    def _load_image(self, path):
        # Retrieve thumbnail path and open image
        thumb_path = ImageThumbCache.instance().get_thumb(path, "medium")
        image = PILPool.get_pil_image(thumb_path)
        # Apply augmentation based on probability if enabled
        if self.use_augmentation and self.augmentation_transform is not None:
            if random.random() < self.augmentation_probability:
                image = self.augmentation_transform(image)
        # Apply final transform (e.g., conversion to tensor, normalization)
        if self.transform:
            image = self.transform(image)
        return image

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
        # Remove the classification head.
        self.backbone.classifier = nn.Identity()
        # Determine the output dimension (typically 1280 for MobileNetV2).
        backbone_output_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1280

        # Map backbone features to the desired embedding dimension.
        self.fc = nn.Linear(backbone_output_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        # Normalize embeddings to unit length.
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# --- Training Loop with Checkpoint Resume and Persistent Train/Val Split ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Define final transform (resize, to tensor, normalization)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset with augmentation enabled (adjust probability as needed)
    full_dataset = RelationDataset(
        mongodb_uri="mongodb://localhost:27017",
        db_name="files_db",
        transform=transform,
        use_augmentation=True,
        augmentation_probability=0.30
    )

    # Create checkpoint directory and handle persistent train/val split.
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    split_path = os.path.join(checkpoint_dir, "train_val_indices.pth")
    total_size = len(full_dataset)
    if os.path.exists(split_path):
        split_data = torch.load(split_path)
        train_indices = split_data['train']
        val_indices = split_data['val']
        print("Loaded existing train/validation split.")
    else:
        train_size = int(0.8 * total_size)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        torch.save({'train': train_indices, 'val': val_indices}, split_path)
        print("Saved new train/validation split.")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Define DataLoaders (using smaller batch size and workers from sample two)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer.
    model = MobileNetEmbeddingNet(embedding_dim=512).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0

    # Resume from checkpoint if available.
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Training loop.
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

        # Validation loop.
        model.eval()
        val_loss = 0.0
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

        # Save an epoch checkpoint.
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch + 1}: {epoch_checkpoint_path}")

        # Save best model if validation loss improves.
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
