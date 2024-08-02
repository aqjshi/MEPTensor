import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import plot_model

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


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set memory growth to prevent TensorFlow from preallocating the entire GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

set_batch_size = 32

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





def build_cnn_model(input_shape, pooling_type='flatten', num_hidden_layers=1, nodes_per_layer=128, num_classes=1, class_weights=None):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Dropout(0.3)  # Regularization with dropout
    ])
    
    if pooling_type == 'flatten':
        model.add(Flatten())
    elif pooling_type == 'global_avg':
        model.add(GlobalAveragePooling3D())
    elif pooling_type == 'global_max':
        model.add(GlobalMaxPooling3D())
    
    for _ in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
        model.add(Dropout(0.5))  # Regularization with dropout
    
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

                     

def tensor_generator(data, labels, batch_size, input_shape):
    while True:
        data_length = len(data)
        indices = np.arange(data_length)
        np.random.shuffle(indices)
        
        for start in range(0, data_length, batch_size):
            end = min(start + batch_size, data_length)
            batch_indices = indices[start:end]
            batch_tensors = []
            batch_labels = labels[batch_indices]
            for tensor_str in data[batch_indices]:
                values = np.fromstring(tensor_str, sep=' ')
                assert len(values) == input_shape[0] ** 3, f"Tensor does not have {input_shape[0] ** 3} values: {len(values)}"
                tensor = values.reshape(input_shape[0], input_shape[0], input_shape[0])
                tensor = tensor[..., np.newaxis]  # Add a channel dimension for CNN
                batch_tensors.append(tensor)
            yield np.array(batch_tensors), np.array(batch_labels)
            
        # Repeat the data
        np.random.shuffle(indices)

def run_model(dataset, labels, input_shape, pooling_type, num_hidden_layers, nodes_per_layer, epochs, model_name, num_classes=1):
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    train_generator = tensor_generator(train_data, train_labels, set_batch_size, input_shape)
    test_generator = tensor_generator(test_data, test_labels, set_batch_size, input_shape)
    steps_per_epoch = len(train_data) // set_batch_size
    validation_steps = len(test_data) // set_batch_size
    X_test, y_test = [], []
    for _ in range(validation_steps):
        X_batch, y_batch = next(test_generator)
        X_test.extend(X_batch)
        y_test.extend(y_batch)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    model = build_cnn_model(input_shape, pooling_type, num_hidden_layers, nodes_per_layer, num_classes, class_weights=class_weight_dict)
    metrics_callback = MetricsCallback(X_test, y_test)
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(X_test, y_test), verbose=1, callbacks=[metrics_callback])

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
    input_shape = (resolution, resolution, resolution, 1) 

    # Load dataset
    dataset = pd.read_csv(filename)

    if 'tensor' not in dataset.columns:
        raise KeyError("The dataset does not contain a 'tensor' column.")

    # Model configurations
    models = [
        {"name": "cnn_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 1},
        {"name": "cnn_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 1},
        # {"name": "cnn_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5},
        {"name": "cnn_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "cnn_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "cnn_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 1}
    ]

    results = []

    for model_config in models:
        model_name = model_config["name"]
        filter_func = model_config["filter"]
        label_column = model_config["label_column"]
        num_classes = model_config["num_classes"]

        filtered_dataset = filter_func(dataset)
        if 'tensor' not in filtered_dataset.columns:
            raise KeyError(f"The filtered dataset for {model_name} does not contain a 'tensor' column.")
        
        tensor_data = filtered_dataset['tensor'].values
        if label_column not in filtered_dataset.columns:
            raise KeyError(f"The filtered dataset for {model_name} does not contain the label column '{label_column}'.")
        
        labels = filtered_dataset[label_column].values

        if model_name == "cnn_chiral":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 128
        elif model_name == "cnn_chiral_01":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 128
        elif model_name == "cnn_chiral_length":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 64
        elif model_name == "cnn_posneg":
            pooling_type = 'flatten'
            hidden_layers = 1
            nodes_per_layer = 128
        elif model_name == "cnn_posneg_all":
            pooling_type = 'flatten'
            hidden_layers = 1
            nodes_per_layer = 256
        elif model_name == "cnn_rs":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 64
        
        result = run_model(tensor_data, labels, input_shape, pooling_type, hidden_layers, nodes_per_layer, epochs, model_name, num_classes)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"cnn_results.csv", index=False)
    print("saved to csv")
    
if __name__ == "__main__":
    main()
