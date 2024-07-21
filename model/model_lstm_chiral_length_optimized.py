import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from argparse import ArgumentParser

PROGRAM_NAME = "model_lstm_chiral_length_optimized.py"
print(PROGRAM_NAME)

def f1_m(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def build_lstm_model(input_shape, pooling_type='flatten', num_hidden_layers=1, nodes_per_layer=128):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    
    if pooling_type == 'flatten':
        model.add(Flatten())
    elif pooling_type == 'global_avg':
        model.add(GlobalAveragePooling1D())
    elif pooling_type == 'global_max':
        model.add(GlobalMaxPooling1D())
    
    for _ in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
    
    model.add(Dense(5, activation='softmax'))  # 5 classes for multi-class classification
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class MetricsCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)

        print(PROGRAM_NAME)
        print(f"Epoch {epoch + 1}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

def process_and_evaluate_model(filename, test_size, input_shape, pooling_type, num_hidden_layers, nodes_per_layer, epochs):
    # Load dataset
    dataset = pd.read_csv(filename)

    # Filter dataset for model_chiral_length
    dataset = dataset[dataset['chiral_length'].isin([0, 1, 2, 3, 4])]

    # Ensure tensor data has 729 values
    def parse_tensor(tensor_str):
        values = np.fromstring(tensor_str, sep=' ')
        assert len(values) == 729, f"Tensor does not have 729 values: {len(values)}"
        return values.reshape(9, 9, 9)

    # Convert tensor data
    tensor_data = np.stack(dataset['tensor'].apply(parse_tensor).values)
    tensor_data = tensor_data.reshape(tensor_data.shape[0], 9, -1)  # Reshape for LSTM

    # Labels
    labels = dataset['chiral_length'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=test_size, random_state=42)

    # Train the LSTM model
    model = build_lstm_model(input_shape, pooling_type, num_hidden_layers, nodes_per_layer)
    metrics_callback = MetricsCallback(X_test, y_test)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, callbacks=[metrics_callback])

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)

    return len(dataset), accuracy, precision, recall, f1

def grid_search(filename, test_size, input_shape, param_grid):
    best_score = 0
    best_params = None
    best_result = None
    
    for params in ParameterGrid(param_grid):
        print(f"{PROGRAM_NAME} Testing with parameters: {params}")
        length, accuracy, precision, recall, f1 = process_and_evaluate_model(
            filename,
            test_size,
            input_shape,
            params['pooling_type'],
            params['num_hidden_layers'],
            params['nodes_per_layer'],
            params['epochs']
        )
        
        score = (accuracy + precision + recall + f1) / 4
        if score > best_score:
            best_score = score
            best_params = params
            best_result = {
                "Length of Filtered Dataset": length,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
    
    print(f"{PROGRAM_NAME} Best parameters: {best_params}")
    return best_result

def main():
    parser = ArgumentParser()
    parser.add_argument("filename", help="The filename of the dataset")
    parser.add_argument("test_size", type=float, help="The test size for the train-test split")
    args = parser.parse_args()

    filename = args.filename
    test_size = args.test_size
    input_shape = (9, 81)  # Adjusted input shape for LSTM

    param_grid = {
        'pooling_type': ['flatten'],
        'num_hidden_layers': [9],
        'nodes_per_layer': [243],
        'epochs': [50]
    }

    best_result = grid_search(filename, test_size, input_shape, param_grid)
    results_df = pd.DataFrame([best_result])
    results_df.to_csv(f"{PROGRAM_NAME}_results.csv", index=False)

if __name__ == "__main__":
    main()
