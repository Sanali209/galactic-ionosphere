import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network that will be shared (Siamese Network)
class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the Siamese network class
class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, input1, input2):
        # Pass both inputs through the shared network
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2


# Contrastive Loss (Optional for similarity-based distance)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute the Euclidean distance between the two outputs
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


# Input size and hyperparameters
input_size = 10  # Number of features
hidden_size = 64
output_size = 32  # Embedding size (can be adjusted)
learning_rate = 0.001
num_epochs = 10

# Initialize the network, loss, and optimizer
base_network = SimpleNetwork(input_size, hidden_size, output_size)
siamese_network = SiameseNetwork(base_network)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(siamese_network.parameters(), lr=learning_rate)

# Dummy dataset (replace with your real data)
# Assume we have 10-dimensional input features
data1 = torch.randn(100, input_size)
data2 = torch.randn(100, input_size)
labels = torch.randint(0, 2, (100,)).float()  # 0 or 1 indicating dissimilar or similar

# Training loop
for epoch in range(num_epochs):
    siamese_network.train()

    # Forward pass
    output1, output2 = siamese_network(data1, data2)

    # Compute loss
    loss = criterion(output1, output2, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
