import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout, Reshape
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import plot_model
from keras import Input

import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid

from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit
from numba import cuda
import os

PROGRAM_NAME = "lstm"
print(PROGRAM_NAME)

# def check_versions_and_gpu():
#     print("TensorFlow version:", tf.__version__)
#     print("Keras version:", keras.__version__)
#     print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# check_versions_and_gpu()


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set memory growth to prevent TensorFlow from preallocating the entire GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
        
def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Calculate the binary cross entropy loss
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        # Apply class weights
        weight_vector = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        weighted_bce = weight_vector * bce
        return tf.reduce_mean(weighted_bce)
    return loss

def f1_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(tf.cast(y_pred, tf.float32))
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    precision = tp / (tp + fp + keras.backend.epsilon())
    recall = tp / (tp + fn + keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
    return tf.reduce_mean(f1)

def build_lstm_model(input_shape, pooling_type='flatten', num_hidden_layers=1, nodes_per_layer=128, num_classes=1, class_weights=None):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    if pooling_type == 'flatten':
        model.add(Flatten())
    elif pooling_type == 'global_avg':
        model.add(GlobalAveragePooling1D())
    elif pooling_type == 'global_max':
        model.add(GlobalMaxPooling1D())
    
    for _ in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
        model.add(Dropout(0.5))
    
    if num_classes > 1:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))

    if class_weights:
        loss = weighted_binary_crossentropy(class_weights)
    else:
        loss = 'binary_crossentropy'

    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy', f1_m])
    return model




class MetricsCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.epoch_metrics = []
        self.conf_matrix = None
        self.class_report = None

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[-1] > 1 else np.where(y_pred > 0.5, 1, 0)

        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred_classes)
        self.class_report = classification_report(self.y_test, y_pred_classes, target_names=['Not Chiral', 'Chiral'])

        self.epoch_metrics.append((epoch + 1, accuracy, precision, recall, f1))

        print(f"Epoch {epoch + 1}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:")
        print(self.conf_matrix)
        print("Classification Report:")
        print(self.class_report)
        
        
def run_model(dataset, input_shape, pooling_type, num_hidden_layers, nodes_per_layer, epochs, model_name, num_classes):
    def parse_tensor(tensor_str):
        values = np.fromstring(tensor_str, sep=' ')
        return values.reshape(input_shape[0], input_shape[0], input_shape[0])

    tensor_data = np.stack(dataset['tensor'].apply(parse_tensor).values)
    tensor_data = tensor_data.reshape(tensor_data.shape[0], input_shape[0], -1)

    labels = dataset[model_name].values

    X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)

    model = build_lstm_model(input_shape, pooling_type, num_hidden_layers, nodes_per_layer, num_classes)
    metrics_callback = MetricsCallback(X_test, y_test)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, callbacks=[metrics_callback])

    y_pred = model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    epoch_metrics = metrics_callback.epoch_metrics

    results = {
        "Model": model_name,
        "Nodes per Layer": nodes_per_layer,
        "Epoch Metrics": epoch_metrics,
        "Confusion Matrix": metrics_callback.conf_matrix.tolist(),
        "Classification Report": metrics_callback.class_report
    }

    return results

def main():
    parser = ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("resolution", type=int)
    parser.add_argument("epochs", type=int)
    args = parser.parse_args()

    filename = args.filename
    resolution = args.resolution
    epochs = args.epochs
    input_shape = (resolution, resolution**2) 
    dataset = pd.read_csv(filename)
    
    
    models = [
        {"name": "lstm_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 1},
        {"name": "lstm_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 1},
        # {"name": "lstm_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5},
        {"name": "lstm_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "lstm_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "lstm_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 1}
    ]
    
    results = []

    for model_config in models:
        model_name = model_config["name"]
        filter_func = model_config["filter"]
        label_column = model_config["label_column"]
        num_classes = model_config["num_classes"]

        filtered_dataset = filter_func(dataset)
        # tensor_data = filtered_dataset['tensor'].values
        
        # labels = filtered_dataset[label_column].values
        if model_name == "lstm_chiral":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 128
        elif model_name == "lstm_chiral_01":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 128
        elif model_name == "lstm_chiral_length":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 64
        elif model_name == "lstm_posneg":
            pooling_type = 'flatten'
            hidden_layers = 1
            nodes_per_layer = 128
        elif model_name == "lstm_posneg_all":
            pooling_type = 'flatten'
            hidden_layers = 1
            nodes_per_layer = 256
        elif model_name == "lstm_rs":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 64
        
        result = run_model(filtered_dataset, input_shape, pooling_type='flatten', num_hidden_layers=2, nodes_per_layer=128, epochs=epochs, model_name=label_column, num_classes=num_classes)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"lstm_results.csv", index=False)
    print("saved to csv")
    
    
if __name__ == "__main__":
    main()
