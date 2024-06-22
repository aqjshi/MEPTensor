import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from tensor import tensor_dataset
from helper import npy_preprocessor

'''
Training format:

index,formula,tensor,rotation0,rotation1,rotation2
079782,O C C C O C C N C H H H H H H H,0.0 0.0 0.0 0.05 0.0 0.01 0.08 0.0 0.0 0.36 1.49 0.0 0.0 0.05 0.13 0.0 0.0 0.0 0.59 2.0 1.02 2.0 2.37 3.63 0.21 2.18 8.92 0.01 0.0 0.0 3.2 4.77 0.0 2.68 6.42 0.9 1.1099999999999999 5.35 2.74 3.25 0.63 3.46 0.01 7.38 0.02 0.12 3.2199999999999998 4.79 0.0 0.71 1.59 0.0 0.17 0.22 7.02 0.8300000000000001 1.72 8.290000000000001 3.4 2.87 1.97 6.159999999999999 0.01 0.0,-0.0,-0.0,-0.0
005049,N C N C C N C N H H H H,0.04 0.0 0.0 0.0 1.3 1.38 0.0 0.0 0.45999999999999996 0.74 0.0 0.0 0.15 0.02 0.0 0.0 0.0 0.0 0.82 1.23 1.23 4.99 1.46 2.06 8.690000000000001 3.57 0.03 0.0 4.03 0.85 0.19 0.64 0.0 0.0 1.67 2.36 0.0 0.0 3.15 4.43 0.13 1.29 0.49 0.0 0.09 6.9799999999999995 2.95 3.5799999999999996 0.0 0.0 1.22 1.71 0.84 4.51 2.49 3.49 7.11 4.890000000000001 0.0 0.0 4.51 1.11 0.0 0.01,-14.99,-12.5,-4979.91


'''


# Load dataset
dataset = pd.read_csv("tensor_dataset.csv")

# Convert all columns to numeric, forcing non-numeric values to NaN and then fill NaN with a placeholder value (e.g., 0)
dataset = dataset.apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert the dataset to a tensor
data = torch.tensor(dataset.values, dtype=torch.float32)

# Assuming last 3 columns are the labels
labels = data[:, -3:]  
# Remaining columns are the input features
data = data[:, :-3]  

# Create a TensorDataset and DataLoader
tensor_dataset = TensorDataset(data, labels)
dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

# Define the model
class SimpleRotationPredictor(nn.Module):
    def __init__(self, input_size):
        super(SimpleRotationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Output 3 values for rotation0, rotation1, and rotation2
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = data.shape[1]
model = SimpleRotationPredictor(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

# Train the model
print("Training model")
train_model(model, dataloader, criterion, optimizer)

# Prediction function
def predict(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions.squeeze().numpy()

# Predict on a new sample (replace new_sample_tensor with your sample tensor)
new_sample_tensor = torch.tensor([data[0].numpy()], dtype=torch.float32)  # Example: first sample
predictions = predict(model, new_sample_tensor)

print(f'Predicted rotations: {predictions}')
