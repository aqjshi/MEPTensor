import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import plot_model

import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid
from argparse import ArgumentParser


from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from wrapper import npy_preprocessor_v4, heat_component, npy_preprocessor_v4_limit


# Check TensorFlow and Keras versions and GPU availability
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create a simple model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1000,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Print model summary
model.summary()

# Generate random data
x_train = np.random.random((10000, 1000))
y_train = np.random.randint(10, size=(10000, 1))
y_train = keras.utils.to_categorical(y_train, 10)

x_test = np.random.random((1000, 1000))
y_test = np.random.randint(10, size=(1000, 1))
y_test = keras.utils.to_categorical(y_test, 10)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Measure the time taken for training
start_time = time.time()

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

end_time = time.time()
print("Training time: {:.2f} seconds".format(end_time - start_time))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
