import os
import torch
import torchvision.transforms as T
from PIL import Image

import pymongo
import timm
import chromadb
from chromadb.config import Settings

# --- ViT-based Embedding Model (Same as in train.py) ---
import torch.nn as nn

class ViTEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
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

# --- Helper function to load and transform an image ---
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# --- Main Inference Function ---
def index_images_and_query(mongodb_uri="mongodb://localhost:27017", db_name="your_db_name", model_path="vit_embedding_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Define the same transforms as used in training.
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
    ])

    # Load the trained model.
    model = ViTEmbeddingNet(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Connect to MongoDB and load image records.
    client = pymongo.MongoClient(mongodb_uri)
    db = client[db_name]
    collection_records = db["collection_records"]

    # Build a dictionary: image_id -> full image path.
    image_records = {}
    for record in collection_records.find({}):
        full_path = os.path.join(record["local_path"], record["name"])
        image_records[str(record["_id"])] = full_path

    # Initialize ChromaDB client.
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    ))
    # Create (or get) a collection to store image embeddings.
    collection = chroma_client.get_or_create_collection("image_embeddings")

    # For each image, compute the embedding and add it to ChromaDB.
    ids = []
    embeddings = []
    metadatas = []
    for image_id, image_path in image_records.items():
        try:
            img = load_image(image_path, transform)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        img_tensor = img.unsqueeze(0).to(device)  # add batch dim
        with torch.no_grad():
            embed = model(img_tensor)
        # Convert embedding to list.
        embed_list = embed.squeeze(0).cpu().tolist()

        ids.append(image_id)
        embeddings.append(embed_list)
        metadatas.append({"image_path": image_path})
        print(f"Indexed image {image_id}")

    # Add embeddings to the ChromaDB collection.
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("All images indexed in ChromaDB.")

    # --- Example Query ---
    # Let’s assume you want to query with an image.
    query_image_path = input("Enter path to query image: ").strip()
    try:
        query_img = load_image(query_image_path, transform)
    except Exception as e:
        print(f"Error loading query image: {e}")
        return

    query_tensor = query_img.unsqueeze(0).to(device)
    with torch.no_grad():
        query_embed = model(query_tensor)
    query_vec = query_embed.squeeze(0).cpu().tolist()

    # Query ChromaDB for the top 5 nearest neighbors.
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5
    )

    print("Query results:")
    for idx, res_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][idx]
        score = results["distances"][0][idx]
        print(f"Rank {idx+1}: Image ID {res_id}, Path: {metadata['image_path']}, Distance: {score}")

if __name__ == "__main__":
    index_images_and_query()
