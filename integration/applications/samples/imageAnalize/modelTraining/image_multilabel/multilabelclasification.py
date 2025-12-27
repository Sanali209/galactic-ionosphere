# Install necessary libraries
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from huggingface_hub import notebook_login

from SLM.files_data_cache.pool import PILPool

# Sample dataset structure
# For demonstration purposes, let's assume you have a CSV file with image paths and corresponding labels
# Make sure to replace this with your actual dataset structure
dataset_csv = """
image_path,label1,label2,label3
/path/to/image1.jpg,1,0,1
/path/to/image2.jpg,0,1,0
/path/to/image3.jpg,1,1,1
# Add more rows as needed
"""

# Save the sample dataset to a CSV file
with open('sample_dataset.csv', 'w') as file:
    file.write(dataset_csv)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        # get pandas columns names without the first column
        self.all_labels = list(self.data.columns[1:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = PILPool.get_pil_image(img_path)
        label = torch.tensor(self.data.iloc[idx, 1:].tolist(), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def onehot_to_labels(self, onehot):
        return [self.all_labels[i] for i, v in enumerate(onehot) if v == 1]

    def labels_to_onehot(self, labels):
        return [1 if l in labels else 0 for l in self.all_labels]


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = CustomDataset(csv_path='sample_dataset.csv', transform=transform)
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


# Define TIMM model
class TIMMModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TIMMModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)



# Instantiate the model
model_name = 'mobilenetv3_large_100'  # Replace with your desired TIMM model
num_classes = 3  # Adjust based on the number of labels in your dataset
model = TIMMModel(model_name, num_classes)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#save model
notebook_login()
model_cfg = dict(labels=['a', 'b', 'c', 'd'])
my_model_name = 'timmmodel'
timm.models.hub.push_to_hf_hub(model, 'my_model_name', model_config=model_cfg)
