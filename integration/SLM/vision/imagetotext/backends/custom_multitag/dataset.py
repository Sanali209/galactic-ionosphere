import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
from collections import Counter
import random
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, db_url, db_name, selected_tags=None, image_size=224, balance_data=True,
                 use_augmentation=True, augmentation_prob=0.5, validation_split=0.2, train=True):
        # Подключение к базе данных MongoDB
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.records = self.db.collection_record
        self.tags = self.db.tags

        # Загружаем все доступные теги из коллекции 'tags'
        all_tags = {tag["name"]: tag["id"] for tag in self.tags.find()}
        if selected_tags:
            # Используем только те теги, которые указаны в selected_tags
            self.selected_tags = {tag: all_tags[tag] for tag in selected_tags if tag in all_tags}
        else:
            # Если selected_tags не передан, используем все теги
            self.selected_tags = all_tags

        # Создаём mapping от тегов к индексам
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.selected_tags.values())}
        self.num_classes = len(self.selected_tags)

        # Загружаем данные
        self.data = []
        for record in self.records.find():
            if "tags" in record and any(tag in self.selected_tags for tag in record["tags"]):
                # Формируем полный путь к изображению
                image_path = os.path.join(record["local_path"], record["filename"])
                self.data.append({
                    "path": image_path,
                    "tags": [self.selected_tags[tag] for tag in record["tags"] if tag in self.selected_tags]
                })

        # Балансировка редких тегов
        if balance_data:
            tag_counts = Counter(tag for item in self.data for tag in item["tags"])
            max_count = max(tag_counts.values())

            balanced_data = []
            for item in self.data:
                path, tags = item["path"], item["tags"]
                multiplier = max_count // min(tag_counts[tag] for tag in tags)
                balanced_data.extend([{"path": path, "tags": tags}] * multiplier)

            self.data = balanced_data

        # Сплит на обучающую и валидационную выборки
        train_data, val_data = train_test_split(self.data, test_size=validation_split, random_state=42)
        self.data = train_data if train else val_data

        # Аугментация (можно отключить)
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")

        # Применяем аугментацию с заданной вероятностью
        if self.use_augmentation and random.random() < self.augmentation_prob:
            image = self.augmentation(image)

        image = self.transform(image)

        # Подготовка меток
        labels = torch.zeros(self.num_classes)
        for tag_id in item["tags"]:
            labels[self.tag_to_idx[tag_id]] = 1

        return image, labels
