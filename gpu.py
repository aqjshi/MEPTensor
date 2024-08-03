import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import time
from numba import cuda

# Import additional libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Check TensorFlow and Keras versions and GPU availability
def check_versions_and_gpu():
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create a simple model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Generate random data
def generate_data(num_samples, num_features, num_classes):
    x = np.random.random((num_samples, num_features))
    y = np.random.randint(num_classes, size=(num_samples, 1))
    y = to_categorical(y, num_classes)
    return x, y

# Clear GPU memory
def clear_gpu_memory():
    print("aint working")

# Main function
def main():
    check_versions_and_gpu()
    
    # Create model
    model = create_model(input_shape=1000, num_classes=10)
    
    # Print model summary
    model.summary()
    
    # Generate training and test data
    x_train, y_train = generate_data(num_samples=10000, num_features=1000, num_classes=10)
    x_test, y_test = generate_data(num_samples=1000, num_features=1000, num_classes=10)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Measure the time taken for training
    start_time = time.time()
    
    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)
    
    
    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

if __name__ == "__main__":
    main()
