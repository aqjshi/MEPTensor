TESTING 
'''
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

'''
Running Tensor.py

python tensor.py [FILENAME] [DENSITY] [PRECISION]
Example

CSV PARSER
python tensor.py xyz.csv 64 2

NPY PARSER

Generate Dataset:
Testing param: python tensor.py qm9_filtered.npy 64 2
Scientific param: python tensor.py qm9_filtered.npy 729 4

python tensor.py [filename] [output_file] [DENSITY] [PRECISION]

Model Training:

python torch_model.py
python mlp_model.py
python cnn_model.py


'''
Training format:

index,formula,tensor,rotation0,rotation1,rotation2
079782,O C C C O C C N C H H H H H H H,0.0 0.0 0.0 0.05 0.0 0.01 0.08 0.0 0.0 0.36 1.49 0.0 0.0 0.05 0.13 0.0 0.0 0.0 0.59 2.0 1.02 2.0 2.37 3.63 0.21 2.18 8.92 0.01 0.0 0.0 3.2 4.77 0.0 2.68 6.42 0.9 1.1099999999999999 5.35 2.74 3.25 0.63 3.46 0.01 7.38 0.02 0.12 3.2199999999999998 4.79 0.0 0.71 1.59 0.0 0.17 0.22 7.02 0.8300000000000001 1.72 8.290000000000001 3.4 2.87 1.97 6.159999999999999 0.01 0.0,-0.0,-0.0,-0.0
005049,N C N C C N C N H H H H,0.04 0.0 0.0 0.0 1.3 1.38 0.0 0.0 0.45999999999999996 0.74 0.0 0.0 0.15 0.02 0.0 0.0 0.0 0.0 0.82 1.23 1.23 4.99 1.46 2.06 8.690000000000001 3.57 0.03 0.0 4.03 0.85 0.19 0.64 0.0 0.0 1.67 2.36 0.0 0.0 3.15 4.43 0.13 1.29 0.49 0.0 0.09 6.9799999999999995 2.95 3.5799999999999996 0.0 0.0 1.22 1.71 0.84 4.51 2.49 3.49 7.11 4.890000000000001 0.0 0.0 4.51 1.11 0.0 0.01,-14.99,-12.5,-4979.91

'''


#how to visulaize nn
PyTorchViz
visualization package. 38

Plan of Execution:

6 properties, 6 models

Chiral Center Existence
01 Chiral Center
Number Chiral Centers
RS classification 1 Chiral
posneg Classification 1 Chiral 
posneg all Classification all Chiral


tensor_dataset_v2.csv 
index, tensor, chiral0, rotation0

RF model on each, migrate to Bluehive, test CNN. 

python [model_name].py [dataset] [output_file] [test_size] [hidden_layer_sizes] [max_iter] 

python chiral_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .05 10 100

#Recommended Traing Hyperparamters
python chiral_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000
python chiral_01_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000

python number_chiral_classifier.py tensor_dataset_v3.csv number_chiral_classifier_pred_v3.csv .2 100 2000

python rs_model.py tensor_dataset_v2.csv rs_model_pred_v2.csv .2 100 2000


python posneg_model.py tensor_dataset_v2.csv posneg_model_pred_v2.csv .2 100 2000

python posneg_all_model.py tensor_dataset_v2.csv chiral_model_pred_v2.csv .2 100 2000






Predictions o

CITATIONS:

https://irvinelab.uchicago.edu/papers/EI12.pdf
https://www.biorxiv.org/content/10.1101/2022.08.24.505155v1.full.pdf

https://www.nature.com/articles/s41377-020-00367-8

https://www.condmatjclub.org/uploads/2013/06/JCCM_JUNE_2013_03.pdf

https://arxiv.org/pdf/2008.01715
