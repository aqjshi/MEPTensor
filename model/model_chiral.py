#filters entire dataset, predicts if chiral or not

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser

print("model_chiral.py")
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("filename", help="The filename of the dataset")
parser.add_argument("test_size", help="The test size for the train-test split")
parser.add_argument("hidden_layer_size", help="The number of neurons in the hidden layer")
parser.add_argument("max_iter", help="The maximum number of iterations for the neural network")
args = parser.parse_args()

# Parse arguments
filename = args.filename
test_size = float(args.test_size)
hidden_layer_size = int(args.hidden_layer_size)
max_iter = int(args.max_iter)

# Load dataset
dataset = pd.read_csv(filename)


# Convert tensors and rotations to numpy arrays
tensor_data = np.stack(dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
index = dataset['index'].values

# Find all chiral_length, if not zero, replace with 1 
chiral_length = dataset['chiral_length'].values 
chiral_length = np.where(chiral_length != 0, 1, 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(tensor_data, chiral_length, test_size=test_size, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the neural network classifier
model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iter, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the chiral lengths on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy, precision, recall, f1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
