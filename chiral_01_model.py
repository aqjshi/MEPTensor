import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser

# Set up argument parser
parser = ArgumentParser()
parser.add_argument("filename", help="The filename of the dataset")
parser.add_argument("output_file", help="The filename of the output file")
parser.add_argument("test_size", help="The test size for the train-test split")
parser.add_argument("hidden_layer_size", help="The number of neurons in the hidden layer")
parser.add_argument("max_iter", help="The maximum number of iterations for the neural network")
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

# Filter the dataset to include only rows where chiral_length is 0 or 1
filtered_dataset = dataset[dataset['chiral_length'].isin([0, 1])]
#print len of filtered dataset
print("Length of filtered dataset:", len(filtered_dataset))
# Convert tensors to numpy arrays
tensor_data = np.stack(filtered_dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
index = filtered_dataset['index'].values
chiral_length = filtered_dataset['chiral_length'].values 

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

# Create DataFrames with the original index, original chiral lengths, and predicted chiral lengths
results = pd.DataFrame({
    'index': index[:len(y_test)],  # Adjust the index to match the length of y_test
    'chiral_length': y_test,
    'predicted_chiral_length': y_pred
})

# Save the results to CSV file
results.to_csv(output_file, index=False)

# Calculate accuracy, precision, recall, f1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
