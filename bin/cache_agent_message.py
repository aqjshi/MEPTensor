import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set GPU configuration to prevent preallocation of GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Fixed encoding dictionary for atom types
atom_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, '0': 5}

# Step 1: File reader function
def read_file(filename):
    return pd.read_csv(filename)

# Step 2: Cleaning function to preprocess agent data
def clean_agent(agent_string):
    clean_str = agent_string.strip("[]").replace("'", "").split(',')
    item = clean_str[0].strip()
    return atom_dict.get(item, 0)  # Default to '0' if the atom type is not recognized

# Step 3: Preprocess tensor data
def preprocess_tensor(tensor_dataset):
    tensor_values = np.array([np.fromstring(x, sep=' ', dtype=np.float32) for x in tensor_dataset['tensor']])
    
    if tensor_values.shape[1] % 729 != 0:
        raise ValueError(f"The number of elements per sample ({tensor_values.shape[1]}) is not divisible by 729.")
    
    time_steps = tensor_values.shape[1] // 729  # Calculate number of time steps
    return tensor_values.reshape(tensor_values.shape[0], time_steps, 729)

# Step 4: Preprocess agent data for LSTM as a sequence
def preprocess_agent_for_lstm(agent_dataset):
    y = []
    for _, row in agent_dataset.iterrows():
        agents = [clean_agent(row[f'agent{i}']) for i in range(32)]
        y.append(agents)
    
    y_encoded = np.array(y)
    return y_encoded.reshape(y_encoded.shape[0], y_encoded.shape[1], 1)

# Step 5: Build LSTM model combining tensor and agent sequences
def build_lstm_model(input_shape, agent_shape, num_classes):
    # Tensor sequence branch
    tensor_input = tf.keras.layers.Input(shape=input_shape, name="tensor_input")
    x1 = LSTM(128, return_sequences=True)(tensor_input)
    
    # Agent sequence branch
    agent_input = tf.keras.layers.Input(shape=agent_shape, name="agent_input")
    x2 = LSTM(128, return_sequences=True)(agent_input)
    
    # Combine tensor and agent sequences
    combined = tf.keras.layers.concatenate([x1, x2])
    x = LSTM(64, return_sequences=False)(combined)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=[tensor_input, agent_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Step 6: Evaluate the performance of each agent's model
def evaluate_performance(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }

# Step 7: Generate and save confusion matrix
# Modify the save_confusion_matrix function to ensure y_true and y_pred are 1D arrays
def save_confusion_matrix(y_true, y_pred, agent_index):
    # Ensure y_true is 1D
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    # Ensure y_pred is 1D (already should be, but just in case)
    y_pred = np.array(y_pred).flatten()
    
    # Determine the classes that were actually present in y_true or y_pred
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Filter atom_dict to only include seen classes
    filtered_atom_dict = {k: atom_dict[k] for k in atom_dict if atom_dict[k] in unique_classes}
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(filtered_atom_dict.keys()), 
                yticklabels=list(filtered_atom_dict.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for Agent {agent_index}')
    plt.savefig(f'images/confusion_matrix_agent_{agent_index}.png')
    plt.close()

# Step 8: Train and evaluate each agent sequentially
def train_and_evaluate_agent(i, X, y_encoded, epochs):
    y_agent = y_encoded[:, i, :]  # Extract the sequence for the i-th agent
    
    # Ensure y_agent has three dimensions (samples, timesteps, features)
    if len(y_agent.shape) == 2:  # If y_agent is 2D (samples, timesteps)
        y_agent = np.expand_dims(y_agent, axis=-1)  # Add a third dimension
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_agent, test_size=0.2, random_state=42)
    
    agent_model = build_lstm_model((X_train.shape[1], X_train.shape[2]), (y_train.shape[1], y_train.shape[2]), num_classes=len(atom_dict))
    agent_model.fit([X_train, y_train], y_train[:, -1], epochs=epochs, validation_data=([X_test, y_test], y_test[:, -1]), verbose=0)
    
    y_pred = np.argmax(agent_model.predict([X_test, y_test]), axis=-1)
    
    save_confusion_matrix(y_test[:, -1], y_pred, i)
    return evaluate_performance(y_test[:, -1], y_pred)

def main():
    parser = ArgumentParser(description="Process tensor and agent CSV files.")
    parser.add_argument("tensor_filename", type=str, help="Filename for the tensor dataset")
    parser.add_argument("agent_filename", type=str, help="Filename for the agent dataset")
    parser.add_argument("epochs", type=int, help="Number of epochs to train the model")
    
    args = parser.parse_args()
    
    tensor_dataset = read_file(args.tensor_filename)
    agent_dataset = read_file(args.agent_filename)
    
    X = preprocess_tensor(tensor_dataset)
    y_encoded = preprocess_agent_for_lstm(agent_dataset)
    
    if len(X) != len(y_encoded):
        raise ValueError(f"Mismatch in number of samples between X and y: {len(X)} vs {len(y_encoded)}")
    
    performance_metrics = {}

    # Train and evaluate each agent sequentially
    for i in range(32):
        metrics = train_and_evaluate_agent(i, X, y_encoded, args.epochs)
        performance_metrics[f"Agent_{i}"] = metrics
    
    with open('performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=4)

if __name__ == "__main__":
    main()
