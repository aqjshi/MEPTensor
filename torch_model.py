import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from tensor import tensor_dataset
from wrapper import npy_preprocessor


'''
Training format:

index,formula,tensor,rotation0,rotation1,rotation2
079782,O C C C O C C N C H H H H H H H,0.0 0.0 0.0 0.05 0.0 0.01 0.08 0.0 0.0 0.36 1.49 0.0 0.0 0.05 0.13 0.0 0.0 0.0 0.59 2.0 1.02 2.0 2.37 3.63 0.21 2.18 8.92 0.01 0.0 0.0 3.2 4.77 0.0 2.68 6.42 0.9 1.1099999999999999 5.35 2.74 3.25 0.63 3.46 0.01 7.38 0.02 0.12 3.2199999999999998 4.79 0.0 0.71 1.59 0.0 0.17 0.22 7.02 0.8300000000000001 1.72 8.290000000000001 3.4 2.87 1.97 6.159999999999999 0.01 0.0,-0.0,-0.0,-0.0
005049,N C N C C N C N H H H H,0.04 0.0 0.0 0.0 1.3 1.38 0.0 0.0 0.45999999999999996 0.74 0.0 0.0 0.15 0.02 0.0 0.0 0.0 0.0 0.82 1.23 1.23 4.99 1.46 2.06 8.690000000000001 3.57 0.03 0.0 4.03 0.85 0.19 0.64 0.0 0.0 1.67 2.36 0.0 0.0 3.15 4.43 0.13 1.29 0.49 0.0 0.09 6.9799999999999995 2.95 3.5799999999999996 0.0 0.0 1.22 1.71 0.84 4.51 2.49 3.49 7.11 4.890000000000001 0.0 0.0 4.51 1.11 0.0 0.01,-14.99,-12.5,-4979.91


'''
# Load dataset
dataset = pd.read_csv("tensor_dataset.csv")
print("Read dataset")

# Extract the relevant columns
formulas = dataset['formula']
tensor_data = dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' '))
print("Finished processing tensor")
rotations = dataset[['rotation0', 'rotation1', 'rotation2']]

# Convert tensors and rotations to numpy arrays
tensor_data = np.stack(tensor_data.values)
rotations = rotations.to_numpy()

tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
rotations = torch.tensor(rotations, dtype=torch.float32)

# Create TensorDataset and DataLoader
print("Creating dataset and dataloader")
dataset = TensorDataset(tensor_data, rotations)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)  # Reduced batch size

print("Finished creating dataset and dataloader")
# Define a minimal neural network model
class RotationPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RotationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)  # Minimal number of neurons
        self.fc2 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
print("Initializing model, loss function, and optimizer")
input_dim = tensor_data.shape[1]
output_dim = rotations.shape[1]
model = RotationPredictor(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Increased learning rate

# Training loop
num_epochs = 1  # Minimal number of epochs
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    for tensors, targets in dataloader:
        # Forward pass
        outputs = model(tensors)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")