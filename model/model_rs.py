#for molecules with 1 chiral center, determine the R or S of the chiral center

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser

print("model_rs.py")
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("filename", help="The filename of the dataset")
parser.add_argument("test_size", type=float, help="The test size for the train-test split")
parser.add_argument("hidden_layer_size", type=int, help="The number of neurons in the hidden layer")
parser.add_argument("max_iter", type=int, help="The maximum number of iterations for the neural network")
args = parser.parse_args()

# Parse arguments
filename = args.filename
test_size = args.test_size
hidden_layer_size = args.hidden_layer_size
max_iter = args.max_iter

# Load dataset
dataset = pd.read_csv(filename)

# Filter the dataset to include only rows where chiral0 is R or S
filtered_dataset = dataset[dataset['chiral0'].isin(['R', 'S'])]
#print len of filtered dataset
print("Length of filtered dataset:", len(filtered_dataset))
# Convert tensors to numpy arrays
tensor_data = np.stack(filtered_dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
index = filtered_dataset['index'].values


# Convert tensors to numpy arrays
tensor_data = np.stack(filtered_dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
chiral_labels = filtered_dataset['chiral0'].apply(lambda x: 1 if x == 'R' else 0).values  # Convert R to 1 and S to 0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(tensor_data, chiral_labels, test_size=test_size, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the neural network classifier
model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iter, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the chiral labels on the test set
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
