import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Flatten, Dense, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser

def build_cnn_model(input_shape):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def process_and_evaluate_model(filename, test_size, input_shape, task_name):
    # Load dataset
    dataset = pd.read_csv(filename)

    # Filter dataset if necessary
    if task_name == "model_chiral_01.py":
        dataset = dataset[dataset['chiral_length'].isin([0, 1])]
    elif task_name == "model_chiral_length.py":
        dataset = dataset[dataset['chiral_length'].isin([0, 1, 2, 3, 4])]
    elif task_name == "model_rs.py":
        dataset = dataset[dataset['chiral0'].isin(['R', 'S'])]
    elif task_name == "model_posneg.py":
        dataset['rotation0'] = np.where(dataset['rotation0'] > 0, 1, 0)

    # Ensure tensor data has 729 values
    def parse_tensor(tensor_str):
        values = np.fromstring(tensor_str, sep=' ')
        assert len(values) == 729, f"Tensor does not have 729 values: {len(values)}"
        return values.reshape(9, 9, 9)

    # Convert tensor data
    tensor_data = np.stack(dataset['tensor'].apply(parse_tensor).values)
    tensor_data = tensor_data[..., np.newaxis]  # Add a channel dimension for CNN

    if task_name == "model_rs.py":
        labels = dataset['chiral0'].apply(lambda x: 1 if x == 'R' else 0).values
    elif task_name == "model_posneg.py":
        labels = dataset['rotation0'].values
    else:
        labels = dataset['chiral_length'].values
        if task_name == "model_chiral.py":
            labels = np.where(labels != 0, 1, 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=test_size, random_state=42)

    # Train the CNN model
    model = build_cnn_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return len(dataset), accuracy, precision, recall, f1, mse, mae

def main():
    parser = ArgumentParser()
    parser.add_argument("filename", help="The filename of the dataset")
    parser.add_argument("test_size", type=float, help="The test size for the train-test split")
    args = parser.parse_args()

    filename = args.filename
    test_size = args.test_size
    input_shape = (9, 9, 9, 1)  # Assuming the tensor is 9x9x9 with a single channel

    task_names = [
        "model_chiral.py",
        "model_chiral_01.py",
        "model_chiral_length.py",
        "model_rs.py",
        "model_posneg.py"
    ]

    results = []

    for task_name in task_names:
        length, accuracy, precision, recall, f1, mse, mae = process_and_evaluate_model(filename, test_size, input_shape, task_name)
        results.append([task_name, length, accuracy, precision, recall, f1, mse, mae])

    results_df = pd.DataFrame(results, columns=["Model", "Length of Filtered Dataset", "Accuracy", "Precision", "Recall", "F1 Score", "MSE", "MAE"])
    results_df.to_csv("model_results.csv", index=False)

if __name__ == "__main__":
    main()
