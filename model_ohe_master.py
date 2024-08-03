import os
import re
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool, cpu_count
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM
from keras.utils import to_categorical

from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit

# Function to iterate through xyz_array
def xyz_iterator(xyz_array):
    for i in range(len(xyz_array)):
        xyz = xyz_array[i]
        yield xyz

# Function to build a MLP model using the Functional API
def build_mlp_model(input_shape, num_hidden_layers=1, nodes_per_layer=128, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    for _ in range(num_hidden_layers):
        x = Dense(nodes_per_layer, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_lstm_model(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = LSTM(128)(inputs)
    outputs = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Function to process molecule data
def process_molecule(args):
    index, inchi, ohe_array, chiral_centers, rotation = args
    chiral0 = chiral_centers[0][1] if chiral_centers else '0'
    return {
        'index': index,
        'inchi': inchi,
        'ohe_array': ohe_array,
        'chiral_length': len(chiral_centers),
        'chiral0': chiral0,
        'rotation0': rotation[0],
        'rotation1': rotation[1],
        'rotation2': rotation[2]
    }

def run_model(tensor_data, labels, input_shape, model_type, num_hidden_layers=1, nodes_per_layer=128, epochs=10, model_name="", num_classes=2):
    X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)
    
    if num_classes > 1:
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
    if model_type == 'mlp':
        model = build_mlp_model(input_shape, num_hidden_layers, nodes_per_layer, num_classes)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1) if num_classes > 1 else (y_pred > 0.5).astype(int)
    elif model_type == 'rf':
        model = build_rf_model()
        model.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1) if num_classes > 1 else y_train)
        y_pred_class = model.predict(X_test.reshape(X_test.shape[0], -1))
    elif model_type == 'lstm':
        model = build_lstm_model(input_shape, num_classes)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1) if num_classes > 1 else (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(np.argmax(y_test, axis=1) if num_classes > 1 else y_test, y_pred_class)
    precision = precision_score(np.argmax(y_test, axis=1) if num_classes > 1 else y_test, y_pred_class, average='weighted')
    recall = recall_score(np.argmax(y_test, axis=1) if num_classes > 1 else y_test, y_pred_class, average='weighted')
    f1 = f1_score(np.argmax(y_test, axis=1) if num_classes > 1 else y_test, y_pred_class, average='weighted')
    return {
        'model_name': model_name,
        'model_type': model_type,
        'nodes_per_layer': nodes_per_layer,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    parser = ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("epochs", type=int)
    args = parser.parse_args()

    filename = args.filename
    epochs = args.epochs
    
    index_array, inchi_array, xyz_array, chiral_centers_array, rotation_array = npy_preprocessor_v4_limit(filename, limit=10000)

    pool = Pool(cpu_count())
    args = [(index_array[i], inchi_array[i], xyz_array[i], chiral_centers_array[i], rotation_array[i]) 
            for i in range(len(index_array))]
    
    results = pool.map(process_molecule, args)
    pool.close()
    pool.join()

    # Filter out None results due to errors
    results = [res for res in results if res is not None]
    # Create DataFrame
    df = pd.DataFrame(results)

    # Convert xyz_array to flattened one-hot encoded arrays
    ohe_arrays = []
    for xyz in xyz_iterator(xyz_array):
        ohe_arrays.append(xyz.flatten())
    
    tensor_data = np.array(ohe_arrays)
    
    models = [
        {"name": "ohe_mlp_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 2, "type": "mlp"},
        {"name": "ohe_rf_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 2, "type": "rf"},
        {"name": "ohe_lstm_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 2, "type": "lstm"},
        {"name": "ohe_mlp_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 2, "type": "mlp"},
        {"name": "ohe_rf_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 2, "type": "rf"},
        {"name": "ohe_lstm_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 2, "type": "lstm"},
        {"name": "ohe_mlp_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5, "type": "mlp"},
        {"name": "ohe_rf_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5, "type": "rf"},
        {"name": "ohe_lstm_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5, "type": "lstm"},
        {"name": "ohe_mlp_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "mlp"},
        {"name": "ohe_rf_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "rf"},
        {"name": "ohe_lstm_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "lstm"},
        {"name": "ohe_mlp_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "mlp"},
        {"name": "ohe_rf_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "rf"},
        {"name": "ohe_lstm_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 2, "type": "lstm"},
        {"name": "ohe_mlp_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 2, "type": "mlp"},
        {"name": "ohe_rf_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 2, "type": "rf"},
        {"name": "ohe_lstm_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 2, "type": "lstm"}
    ]

    results = []
    for model_config in models:
        model_name = model_config["name"]
        filter_func = model_config["filter"]
        label_column = model_config["label_column"]
        num_classes = model_config["num_classes"]
        model_type = model_config["type"]

        filtered_dataset = filter_func(df)
        if 'ohe_array' not in filtered_dataset.columns:
            raise KeyError(f"The filtered dataset for {model_name} does not contain an 'ohe_array' column.")
        
        tensor_data = np.stack(filtered_dataset['ohe_array'].values)
        if label_column not in filtered_dataset.columns:
            raise KeyError(f"The filtered dataset for {model_name} does not contain the label column '{label_column}'.")
        
        labels = filtered_dataset[label_column].values

        input_shape = tensor_data.shape[1:]
        for nodes_per_layer in [32]:
            result = run_model(tensor_data, labels, input_shape, model_type, 2, nodes_per_layer, epochs, model_name, num_classes)
            results.append(result)

    # Output results
    epoch = 0
    for result in results:
        epoch +=1
        print(f"Model: {result['model_name']}, Epoch: {epoch}, Nodes per layer: {result['nodes_per_layer']}, "
              f"Accuracy: {result['accuracy']}, Precision: {result['precision']}, "
              f"Recall: {result['recall']}, F1 Score: {result['f1_score']}")

if __name__ == "__main__":
    main()
