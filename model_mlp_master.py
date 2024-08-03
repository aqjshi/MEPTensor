import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from argparse import ArgumentParser

set_batch_size = 32

def f1_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

def build_mlp_model(input_shape, num_hidden_layers=1, nodes_per_layer=128, num_classes=1):
    model = Sequential([
        Flatten(input_shape=input_shape),
    ])
    
    for _ in range(num_hidden_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
    
    if num_classes > 1:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))

    loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy', f1_m])
    return model

class MetricsCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[-1] > 1 else np.where(y_pred > 0.5, 1, 0)

        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted', zero_division=0)

        self.epoch_metrics.append((epoch + 1, accuracy, precision, recall, f1))

        print(f"Epoch {epoch + 1}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

def tensor_generator(data, labels, batch_size, input_shape):
    while True:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_tensors = []
            batch_labels = labels[start:end]
            for tensor_str in data[start:end]:
                values = np.fromstring(tensor_str, sep=' ')
                assert len(values) == np.prod(input_shape), f"Tensor does not have {np.prod(input_shape)} values: {len(values)}"
                tensor = values.reshape(input_shape)
                batch_tensors.append(tensor)
            yield np.array(batch_tensors), np.array(batch_labels)

def run_model(dataset, labels, input_shape, num_hidden_layers, nodes_per_layer, epochs, model_name, num_classes=1):
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    train_generator = tensor_generator(train_data, train_labels, set_batch_size, input_shape)
    test_generator = tensor_generator(test_data, test_labels, set_batch_size, input_shape)

    steps_per_epoch = len(train_data) // set_batch_size
    validation_steps = len(test_data) // set_batch_size

    model = build_mlp_model(input_shape, num_hidden_layers, nodes_per_layer, num_classes)
    metrics_callback = MetricsCallback(next(test_generator)[0], test_labels[:set_batch_size])
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=test_generator, validation_steps=validation_steps, verbose=1, callbacks=[metrics_callback])

    epoch_metrics = metrics_callback.epoch_metrics

    results = {
        "Model": model_name,
        "Nodes per Layer": nodes_per_layer,
        "Epoch Metrics": epoch_metrics
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
    input_shape = (resolution, resolution, resolution)  # Assuming the tensor is 9x9x9

    dataset = pd.read_csv(filename)

    if 'tensor' not in dataset.columns:
        raise KeyError("The dataset does not contain a 'tensor' column.")

    models = [
        {"name": "mlp_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 1},
        {"name": "mlp_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 1},
        {"name": "mlp_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5},
        {"name": "mlp_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "mlp_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
        {"name": "mlp_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 1}
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

        for nodes_per_layer in [32, 64]:
            result = run_model(tensor_data, labels, input_shape, 2, nodes_per_layer, epochs, model_name, num_classes)
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"mlp_results.csv", index=False)

if __name__ == "__main__":
    main()
