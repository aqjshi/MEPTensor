import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from voxel_preprocessor_v2 import read_voxel_data, preprocess_limited_voxels, preprocess_data, voxel_to_image, extract_patches, process_and_visualize

from torchviz import make_dot



def visualize_model_architecture(model, input_size):
    # Create a dummy input tensor with the given input size
    dummy_input = torch.randn(input_size).to(device)
    
    # Pass the dummy input through the model
    output, _, _ = model(dummy_input)
    
    # Generate the computation graph
    graph = make_dot(output, params=dict(model.named_parameters()))
    
    # Render the graph
    graph.render("model_architecture", format="png")
    print("Model architecture has been saved as 'model_architecture.png'")

def voxel_to_image(voxel_data):
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.cpu().numpy()  # Move to CPU and then convert to numpy array
    
    r_channel = voxel_data[0::3].reshape((9, 9, 9))  # Red channel
    g_channel = voxel_data[1::3].reshape((9, 9, 9))  # Green channel
    b_channel = voxel_data[2::3].reshape((9, 9, 9))  # Blue channel

    # Normalize the channels to [0, 255]
    r_channel = (r_channel * 255).astype(np.uint8)
    g_channel = (g_channel * 255).astype(np.uint8)
    b_channel = (b_channel * 255).astype(np.uint8)

    # Flatten the 3D grid into a 2D 27x27 image
    r_flat = r_channel.flatten().reshape(27, 27)
    g_flat = g_channel.flatten().reshape(27, 27)
    b_flat = b_channel.flatten().reshape(27, 27)

    # Stack the channels to form a 27x27 RGB image
    image = np.stack((r_flat, g_flat, b_flat), axis=-1)
    
    return image


def extract_class_from_filename(filename, classification_task):
    # Assuming the filename format is  {index}${chiral_length}${rs}${posneg}.txt
    parts = filename.split('$')  # Split by '$'
    
    # Extract the relevant parts from the filename
    chiral_length = int(parts[1])  # chiral_length is the second element in the filename
    rs = parts[2]  # rs is the third element in the filename
    posneg = int(parts[3].split('.')[0])  # posneg is the fourth element, remove .txt

    if classification_task == 0:
        # Task 0: Cast chiral_length > 0 to 1, keep 0
        class_label = 1 if chiral_length > 0 else 0

    elif classification_task == 1:
        # Task 1: Same as task 0, cast chiral_length > 0 to 1, keep 0
        class_label = 1 if chiral_length > 0 else 0

    elif classification_task == 2:
        # Task 2: Get chiral_length without casting
        class_label = chiral_length

    elif classification_task == 3:
        # Task 3: Get rs (R/S configuration)
        class_label = rs  # Either 'R' or 'S'

    elif classification_task == 4:
        # Task 4: Get posneg, return 1 if positive, -1 if negative
        class_label = 1 if posneg >= 0 else -1

    elif classification_task == 5:
        # Task 5: Same as task 4, get posneg, return 1 if positive, -1 if negative
        class_label = 1 if posneg >= 0 else -1

    return class_label



# Task 1: Chiral Center Existence {0, 1, . . . , 8} cast into 1 if above zero else zero
# Task 2: 0 vs. 1 Chiral Center {0, 1} filter in chiral_lenth < 2 no cast
# Task 3: Number of Chiral Centers {0, 1, . . . , 8} no cast
# Task 4: R vs. S {1} filter in chiral_lenth == 1
# Task 5: + vs. - {1} filter in chiral_lenth == 1
# Task 6: + vs. - for all {0, 1, . . . , 8} no filter 
    
def preprocess_data_with_labels(data_dir, num_voxels, classification_task):
    voxel_data = preprocess_data(data_dir, num_voxels)
    filtered_voxel_data = []
    labels = []
    unique_classes = set()
    
    files = sorted(os.listdir(data_dir))[:num_voxels]
    
    for i, file in enumerate(files):
        # Task 1: Chiral Center Existence {0, 1}
        if classification_task == 0:
            chiral_length = int(file.split('$')[1])
            class_label = 1 if chiral_length > 0 else 0
        
        # Task 2: 0 vs. 1 Chiral Center {0, 1}
        elif classification_task == 1:
            chiral_length = int(file.split('$')[1])
            if chiral_length >= 2:
                continue  # Skip files where chiral length >= 2
            class_label = chiral_length  # 0 if no chiral center, 1 if exactly one
        
        # Task 3: Number of Chiral Centers {0, 1, ..., 8}
        elif classification_task == 2:
            chiral_length = int(file.split('$')[1])
            class_label = chiral_length  # Use the exact number of chiral centers as the label
        
        # Task 4: R vs. S {1}
        elif classification_task == 3:
            chiral_length = int(file.split('$')[1])
            if chiral_length != 1:
                continue  # Skip files where chiral length != 1
            rs = file.split('$')[2]  # Assuming 'R' or 'S' is present in this part
            class_label = rs  # 'R' or 'S'
        
        # Task 5: + vs. - {1}
        elif classification_task == 4:
            chiral_length = int(file.split('$')[1])
            if chiral_length != 1:
                continue  # Skip files where chiral length != 1
            posneg = int(file.split('$')[3].split('.')[0])
            class_label = posneg  # '+' or '-'
        
        # Task 6: + vs. - for all {0, 1, ..., 8}
        elif classification_task == 5:
            posneg = int(file.split('$')[3].split('.')[0])
            class_label = posneg  # '+' or '-' for all chiral lengths
        
        # Add the class label and corresponding voxel data
        filtered_voxel_data.append(voxel_data[i])  # Append voxel data that passes the filters
        labels.append(class_label)
        unique_classes.add(class_label)
    
    # Map unique classes to numeric labels
    class_to_label = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    numeric_labels = [class_to_label[label] for label in labels]
    
    return filtered_voxel_data, numeric_labels, class_to_label





# Function to create datasets
def create_datasets(voxel_data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(voxel_data, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test



# Feature extraction function
def extract_features_from_patch(patch):
    mean_r = np.mean(patch[:, :, 0])
    mean_g = np.mean(patch[:, :, 1])
    mean_b = np.mean(patch[:, :, 2])
    
    std_r = np.std(patch[:, :, 0])
    std_g = np.std(patch[:, :, 1])
    std_b = np.std(patch[:, :, 2])
    
    features = np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b])
    return features


def extract_features_from_voxel(voxel_data):
    patches = extract_patches(voxel_data)
    all_features = [extract_features_from_patch(patch) for patch in patches]
    return np.array(all_features)


# Function to visualize patches
def visualize_patches(patches, title="Patches"):
    fig, axes = plt.subplots(1, len(patches), figsize=(15, 5))
    for i, patch in enumerate(patches):
        axes[i].imshow(patch)
        axes[i].axis('off')
        axes[i].set_title(f'Patch {i+1}')
    plt.suptitle(title)
    plt.show()


# Function to visualize embeddings
def visualize_embeddings(embeddings, method="PCA"):
    if method == "PCA":
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
    elif method == "TSNE":
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title(f"Embeddings Visualization using {method}")
    plt.show()


# Function to visualize attention weights
def visualize_attention_weights(attention_weights):
    sns.heatmap(attention_weights, annot=True, cmap="viridis")
    plt.title("Attention Weights")
    plt.show()


# Patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, num_patches):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.flatten = nn.Flatten(start_dim=2)  # Flatten each patch separately
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
    def forward(self, patches):
        # print(f"PatchEmbedding - patches shape: {patches.shape}")
        # Manually calculate the correct view shape
        patch_dim = self.projection.in_features  # This should be patch_size * patch_size * in_channels
        patches = patches.view(-1, self.num_patches, patch_dim)  # Reshape for multiple patches
        # print(f"PatchEmbedding - reshaped patches shape: {patches.shape}")
        patches = self.flatten(patches)  # Flatten within each patch
        # print(f"PatchEmbedding - flattened patches shape: {patches.shape}")
        embeddings = self.projection(patches)
        # print(f"PatchEmbedding - embeddings shape: {embeddings.shape}")
        return embeddings





class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(d_model, nhead, **kwargs)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # print(f"CustomTransformerEncoderLayer - src shape before attention: {src.shape}")
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # print(f"CustomTransformerEncoderLayer - src shape after attention: {src.shape}")
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights
    

# Positional encoding layer
# Positional encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.create_positional_encoding(embed_dim, num_patches)
    
    def create_positional_encoding(self, embed_dim, num_patches):
        pe = torch.zeros(1, num_patches, embed_dim)  # Ensure shape matches embeddings
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # print(f"PositionalEncoding - input x shape: {x.shape}")
        positional_encoding = self.positional_encoding.expand(x.size(0), -1, -1).to(x.device)
        # print(f"PositionalEncoding - positional_encoding shape: {positional_encoding.shape}")
        x = x + positional_encoding  # Add positional encoding
        # print(f"PositionalEncoding - output x shape: {x.shape}")
        return x


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, num_patches, patch_size, in_channels, embed_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim, num_patches)
        self.positional_encoding = PositionalEncoding(embed_dim, num_patches)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, 8)  # Assuming 8 classes for 8 octants
    
    def forward(self, patches):
        embeddings = self.patch_embedding(patches)
        embeddings = self.positional_encoding(embeddings)  # This should now work correctly
        
        attention_weights = []
        x = embeddings
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        classification_output = self.classifier(x.mean(dim=1))  # Global average pooling and classification
        return classification_output, embeddings, attention_weights





# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for patches, labels in dataloader:
            optimizer.zero_grad()
            # print(f"Training - patches shape: {patches.shape}")
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")


# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for patches, labels in dataloader:
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")



# Function to visualize predictions
def visualize_predictions(voxel_data, labels, predictions, class_to_label, num_images=5):
    label_to_class = {v: k for k, v in class_to_label.items()}  # Reverse the class_to_label mapping
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        voxel = voxel_data[i]
        label = labels[i].item()
        prediction = predictions[i].item()
        
        image = voxel_to_image(voxel)
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"True: {label_to_class[label]}\nPred: {label_to_class[prediction]}")
    
    plt.suptitle("Model Predictions")
    plt.savefig('model_predictions.png')
    plt.close()



# Evaluation function with prediction visualization
# Evaluation function with prediction visualization, F1 score, and confusion matrix
def evaluate_and_visualize_model(model, dataloader, criterion, voxel_data, y_test, class_to_label, num_images=5):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for patches, labels in dataloader:
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

    # Calculate F1 score
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"F1 Score (Weighted): {f1:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_to_label.keys()), yticklabels=list(class_to_label.keys()))
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()


    # Visualize predictions
    visualize_predictions(voxel_data=X_test, labels=y_test, predictions=predictions, class_to_label=class_to_label, num_images=num_images)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Example usage
if __name__ == "__main__":
    data_dir = "images/"  # Replace with your data directory
    num_voxels = 121416  # Example number of voxels to preprocess
    clasification_task = 3
    print(f"Classification Task: {clasification_task}")

    # Preprocess the data and generate labels
    voxel_data, labels, class_to_label = preprocess_data_with_labels(data_dir, num_voxels, clasification_task)
    print(f"Class to label mapping: {class_to_label}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = create_datasets(voxel_data, labels)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    # Convert to PyTorch tensors and create dataloaders
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the Transformer model with appropriate parameters
    patch_size = 9
    in_channels = 3  # RGB channels
    embed_dim = 64
    num_heads = 8
    num_layers = 6
    num_patches = 9 # For a 27x27 image split into 9 patches

    model = TransformerModel(num_patches=num_patches, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model

    # Visualize the model architecture before training
    visualize_model_architecture(model, (1, num_patches, patch_size * patch_size * in_channels))  # Corrected input size

    train_model(model, train_loader, criterion, optimizer, num_epochs=100)


    # Evaluate the model on the test set
    evaluate_and_visualize_model(model, test_loader, criterion, X_test, y_test, class_to_label, num_images=20)
