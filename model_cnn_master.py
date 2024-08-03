import os
import re
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (LSTM, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, 
                          Conv3D, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D, 
                          BatchNormalization, Dropout)
import keras
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from argparse import ArgumentParser

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

def weighted_loss(class_weights, loss_fn):
    class_weights_tensor = tf.constant([class_weights[i] for i in sorted(class_weights.keys())], dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weight_vector = tf.gather(class_weights_tensor, tf.argmax(y_true, axis=-1))
        return tf.reduce_mean(weight_vector * loss_fn(y_true, y_pred))
    return loss



def f1_m(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.round(tf.cast(y_pred, tf.float32))
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return tf.reduce_mean(f1)

def build_cnn_model(input_shape, pooling_type='flatten', num_hidden_layers=1, nodes_per_layer=128, num_classes=2, class_weights=None):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2), padding='same'),
        Conv3D(64, (3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2), padding='same'),
        Dropout(0.3),
        Flatten() if pooling_type == 'flatten' else 
        GlobalAveragePooling3D() if pooling_type == 'global_avg' else 
        GlobalMaxPooling3D()
    ])
    for _ in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    loss = weighted_loss(class_weights, keras.losses.CategoricalCrossentropy())
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])
    return model
        
class MetricsCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test, self.y_test = X_test, y_test
        self.epoch_metrics = []
        self.conf_matrix = None
        self.class_report = None

    def on_epoch_end(self, epoch, logs=None):
        # BINARY Label Ad Hoc casting
        # 1 is always the positive or 'first' one, for cnn chiral: chiral; for cnn_chiral_01: chiral, posneg: 1; posneg_all: 1; rs: r
        # 0 is always the negative or 'sub' one, for cnn chiral: not chiral; for cnn_chiral_01: not chiral, posneg: -1; posneg_all: -1; rs: s
        # chiral_length is not binary
        
        y_pred = self.model.predict(self.X_test)
        y_pred_classes, y_test_classes = np.argmax(y_pred, axis=1), np.argmax(self.y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        self.epoch_metrics.append((epoch + 1, accuracy, precision, recall, f1))
        
        # Confusion matrix and classification report with binary labeling
        self.conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Custom target names based on binary label ad hoc casting
        if len(np.unique(y_test_classes)) == 2:
            target_names = ['Not Chiral', 'Chiral']
        else:
            target_names = [f'Class {i}' for i in range(len(np.unique(y_test_classes)))]

        self.class_report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)

        # Print metrics and reports
        print(f"Epoch {epoch + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print("Confusion Matrix:")
        print(self.conf_matrix)
        print("Classification Report:")
        print(self.class_report)


def tensor_generator(data, labels, batch_size, input_shape):
    while True:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_indices = indices[start:end]
            batch_tensors = [np.fromstring(data[i], sep=' ').reshape(*input_shape, 1) for i in batch_indices]
            yield np.array(batch_tensors), np.array(labels[batch_indices])


def run_model(dataset, labels, input_shape, pooling_type, num_hidden_layers, nodes_per_layer, epochs, model_name, num_classes=2):
    labels = to_categorical(labels, num_classes=num_classes)
    
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    train_generator = tensor_generator(train_data, train_labels, set_batch_size, input_shape)
    test_generator = tensor_generator(test_data, test_labels, set_batch_size, input_shape)
    steps_per_epoch = int(len(train_data) // set_batch_size /100)
    validation_steps = int(len(test_data) // set_batch_size /100)
    X_test, y_test = [], []
    for _ in range(validation_steps):
        X_batch, y_batch = next(test_generator)
        X_test.extend(X_batch)
        y_test.extend(y_batch)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(train_labels, axis=1)), y=np.argmax(train_labels, axis=1))
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

    # BINARY Label Ad Hoc casting
    # 1 is always the positive or 'first' one, for cnn chiral: chiral; for cnn_chiral_01: chiral, posneg: 1; posneg_all: 1;rs: r
    # 0 is always the negative or 'sub' one, for cnn chiral: not chiral; for cnn_chiral_01: not chiral, posneg: -1; posneg_all: -1;rs: s
    # chiral_length is not binary

    models = [
    {
        "name": "cnn_chiral",
        "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)),
        "label_column": "chiral",
        "num_classes": 2
    },
    {
        "name": "cnn_chiral_01",
        "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']),
        "label_column": "chiral_length_01",
        "num_classes": 2
    },
    {
        "name": "cnn_chiral_length",
        "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])],
        "label_column": "chiral_length",
        "num_classes": 5
    },
    {
        "name": "cnn_posneg",
        "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)),
        "label_column": "rotation0_binary",
        "num_classes": 2
    },
    {
        "name": "cnn_posneg_all",
        "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)),
        "label_column": "rotation0_binary",
        "num_classes": 2
    },
    {
        "name": "cnn_rs",
        "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)),
        "label_column": "chiral_binary",
        "num_classes": 2
    }
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