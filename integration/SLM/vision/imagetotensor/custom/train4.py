import os
import random
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import timm  # Make sure to install timm via pip
import os
import random
import pymongo
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from SLM.files_data_cache.thumbnail import ImageThumbCache
from SLM.appGlue.core import Allocator
from SLM.files_data_cache.pool import PILPool


# Assuming SLM imports are correct and available in the environment


# Ensure that PILPool and ImageThumbCache are imported or defined in your environment

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
        self.training_samples = []  # Expanded samples with explicit negatives

        # Load dataset from MongoDB
        self.prepare_items()

    def prepare_items(self):
        """Loads data from MongoDB and prepares positive/negative pairs."""
        client = pymongo.MongoClient(self.mongodb_uri)  # Connect to MongoDB
        db = client[self.db_name]

        # Load image records from collection_records
        collection_records = db["collection_records"]
        for record in tqdm(collection_records.find({"item_type": "FileRecord"}), desc="Loading images"):
            full_path = os.path.join(record["local_path"], record["name"])
            self.image_paths[record["_id"]] = full_path

        # Define relation types for positive and negative pairs
        self.positive_types = {"similar", "near_dub", "similar_style", "some_person", "some_image_set", "other"
            ,"manual"}
        self.negative_type = "wrong"

        # Load relation records from relation_records collection
        relation_records = list(db["relation_records"].find({"type": "similar_search"}))
        for rec in tqdm(relation_records, desc="Loading relations"):
            sub_type = rec.get("sub_type", "").lower()
            from_id, to_id = rec.get("from_id"), rec.get("to_id")
            if sub_type in self.positive_types and from_id in self.image_paths and to_id in self.image_paths:
                self.positive_pairs.append((from_id, to_id))
            elif sub_type == self.negative_type and from_id in self.image_paths and to_id in self.image_paths:
                self.anchor_to_negatives.setdefault(from_id, []).append(to_id)

        # Store all image IDs (useful for sampling random negatives)
        self.all_image_ids = list(self.image_paths.keys())
        print(f"Found {len(self.positive_pairs)} positive pairs.")
        
        # Expand training samples: create N samples for each positive pair with N explicit negatives
        for anchor_id, positive_id in self.positive_pairs:
            explicit_negatives = self.anchor_to_negatives.get(anchor_id, [])
            if explicit_negatives:
                # Create one training sample for each explicit negative
                for negative_id in explicit_negatives:
                    self.training_samples.append((anchor_id, positive_id, negative_id))
            else:
                # No explicit negatives: create one sample with None (will use random negative)
                self.training_samples.append((anchor_id, positive_id, None))
        
        print(f"Expanded to {len(self.training_samples)} training samples "
              f"(from {len(self.positive_pairs)} positive pairs).")
        print(f"Found {sum(len(negs) for negs in self.anchor_to_negatives.values())} explicit negative relations.")
        client.close()  # Close the connection

    def __len__(self):
        return len(self.training_samples)

    def _load_image(self, path):
        # Retrieve thumbnail path (assuming ImageThumbCache and PILPool are defined)
        path = ImageThumbCache.instance().get_thumb(path, "medium")
        image = PILPool.get_pil_image(path)
        # Apply augmentation based on probability if enabled
        if self.use_augmentation and self.augmentation_transform is not None:
            if random.random() < self.augmentation_probability:
                image = self.augmentation_transform(image)
        # Apply final transform (e.g., conversion to tensor, normalization)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        # Get the pre-assigned training sample (with explicit negative if available)
        anchor_id, positive_id, negative_id = self.training_samples[idx]
        anchor_path = self.image_paths[anchor_id]
        positive_path = self.image_paths[positive_id]

        # If no explicit negative was assigned, select a random one
        if negative_id is None:
            negative_id = random.choice(self.all_image_ids)
            # Ensure negative is not the same as anchor or positive
            while negative_id == anchor_id or negative_id == positive_id:
                negative_id = random.choice(self.all_image_ids)

        # Load images
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(self.image_paths[negative_id])

        return {"anchor": anchor_img, "positive": positive_img, "negative": negative_img}


# --- ViT-based Embedding Model ---
class ViTEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(ViTEmbeddingNet, self).__init__()
        # Load a pretrained ViT model.
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        # Remove the classification head.
        self.backbone.reset_classifier(0)
        backbone_output_dim = self.backbone.num_features  # feature dim from ViT

        # Add a linear layer to map features to the desired embedding dimension.
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

    # Define transforms (adjust as needed)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset
    full_dataset = RelationDataset(mongodb_uri="mongodb://localhost:27017",
                                   db_name="files_db",
                                   transform=transform, use_augmentation=True, augmentation_probability=0.15)

    # Create a directory to save checkpoints and split indices.
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # --- Persistent Train/Validation Split ---
    split_path = os.path.join(checkpoint_dir, "train_val_indices.pth")
    total_size = len(full_dataset)
    if os.path.exists(split_path):
        split_data = torch.load(split_path)
        train_indices = split_data['train']
        val_indices = split_data['val']
        print("Loaded existing train/validation split.")
    else:
        train_size = int(0.9 * total_size)
        # Shuffle indices using a fixed seed if you need extra determinism
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        torch.save({'train': train_indices, 'val': val_indices}, split_path)
        print("Saved new train/validation split.")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # Initialize model, loss, optimizer.
    model = ViTEmbeddingNet().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0

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
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    Allocator.init_modules()

    train_model()
