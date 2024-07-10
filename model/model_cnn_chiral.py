import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D
from tensorflow.keras.utils import to_categorical
from argparse import ArgumentParser

# Set up argument parser
parser = ArgumentParser()
parser.add_argument("filename", help="The filename of the dataset")
parser.add_argument("output_file", help="The filename of the output file")
parser.add_argument("test_size", help="The test size for the train-test split")
parser.add_argument("hidden_layer_size", help="The number of neurons in the hidden layer")
parser.add_argument("max_iter", help="The maximum number of epochs for the neural network")
args = parser.parse_args()

# Parse arguments
filename = args.filename
output_file = args.output_file
test_size = float(args.test_size)
hidden_layer_size = int(args.hidden_layer_size)
max_iter = int(args.max_iter)

# Load dataset
dataset = pd.read_csv(filename)

# Print available columns for debugging
print("Available columns in dataset:", dataset.columns)

# Convert tensors and rotations to numpy arrays
tensor_data = np.stack(dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
index = dataset['index'].values

# Find all chiral_length, if not zero, replace with 1
chiral_length = dataset['chiral_length'].values
chiral_length = np.where(chiral_length != 0, 1, 0)

# Reshape the tensor data for CNN input
# Assuming tensor_data contains images or similar data, reshape accordingly.
# Here we assume the tensor is reshaped into (cuberoot_density, cuberoot_density, cuberoot_density)
cuberoot_density = int(np.cbrt(tensor_data.shape[1]))  # Assuming the tensor data is cubic
tensor_data = tensor_data.reshape(-1, cuberoot_density, cuberoot_density, cuberoot_density, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(tensor_data, chiral_length, test_size=test_size, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = X_train.reshape(-1, cuberoot_density * cuberoot_density * cuberoot_density)
X_test = X_test.reshape(-1, cuberoot_density * cuberoot_density * cuberoot_density)

X_train_scaled = scaler.fit_transform(X_train).reshape(-1, cuberoot_density, cuberoot_density, cuberoot_density, 1)
X_test_scaled = scaler.transform(X_test).reshape(-1, cuberoot_density, cuberoot_density, cuberoot_density, 1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define the CNN model
model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(cuberoot_density, cuberoot_density, cuberoot_density, 1)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(hidden_layer_size, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=max_iter, batch_size=32, validation_split=0.1, verbose=1)

# Predict the chiral lengths on the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Create DataFrames with the original index, original chiral lengths, and predicted chiral lengths for the test set
results = pd.DataFrame({
    'index': index[:len(y_test_classes)],  # Adjust the index to match the length of y_test
    'chiral_length': y_test_classes,
    'predicted_chiral_length': y_pred_classes
})

# Save the results to CSV file
results.to_csv(output_file, index=False)

# Calculate accuracy, precision, recall, f1 score
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
