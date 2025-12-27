import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import os
from dataset import ImageDataset

# === Параметры ===
DB_URL = "mongodb://localhost:27017/"
DB_NAME = "files_db"
SELECTED_TAGS = ["cat", "dog", "car"]  # Можете указать нужные теги
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
MODEL_PATH = "vit_multilabel.pth"
BEST_MODEL_PATH = "best_vit_multilabel.pth"
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5
VALIDATION_SPLIT = 0.2  # 20% данных на валидацию

# === Датасет ===
train_dataset = ImageDataset(DB_URL, DB_NAME, SELECTED_TAGS, image_size=IMAGE_SIZE,
                             use_augmentation=USE_AUGMENTATION, augmentation_prob=AUGMENTATION_PROB,
                             validation_split=VALIDATION_SPLIT, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageDataset(DB_URL, DB_NAME, SELECTED_TAGS, image_size=IMAGE_SIZE,
                           use_augmentation=False, augmentation_prob=0,
                           validation_split=VALIDATION_SPLIT, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загружаем старую модель, если есть ===
if os.path.exists(MODEL_PATH):
    old_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=train_dataset.num_classes)
    old_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=train_dataset.num_classes)
    model.load_state_dict(old_model.state_dict(), strict=False)
else:
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=train_dataset.num_classes)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Обучение ===
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Сохранение лучшей модели
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Val Loss: {best_val_loss:.4f}")

# Сохранение финальной модели
torch.save(model.state_dict(), MODEL_PATH)
print("Final model saved!")
