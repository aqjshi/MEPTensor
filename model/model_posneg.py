#for molecules with one chiral center, determine the pos or neg of rotation
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparse import ArgumentParser

print("model_posneg.py")
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


tensor_data = np.stack(dataset['tensor'].apply(lambda x: np.fromstring(x, sep=' ')).values)
rotations = dataset['rotation0'].values
rotations = np.where(rotations > 0, 1, 0)

index = dataset['index'].values

# Print length of filtered dataset
print("Length of filtered dataset:", len(dataset))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(tensor_data, rotations, test_size=test_size, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the neural network classifier
model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iter, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the rotation labels on the test set
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
