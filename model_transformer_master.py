
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

from transformers import TFBertModel, TFGPT2Model, TFT5Model, BertTokenizer, GPT2Tokenizer, T5Tokenizer
# Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Encoder Layer
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = models.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer Model
def build_transformer_model(input_shape, num_heads, num_layers, d_model, dff, num_classes, max_len):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]))(inputs)
    x = PositionalEncoding(d_model, max_len)(x)

    for _ in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    if num_classes > 1:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    return model

# Example usage with BERT
def build_bert_model(input_shape, num_classes, max_len):
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
    x = layers.GlobalAveragePooling1D()(bert_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    if num_classes > 1:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    return model

# Example usage with GPT-3
def build_gpt3_model(input_shape, num_classes, max_len):
    gpt3_model = TFGPT2Model.from_pretrained("gpt2")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    gpt3_output = gpt3_model(input_ids)[0]
    x = layers.GlobalAveragePooling1D()(gpt3_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    if num_classes > 1:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    
    model = Model(inputs=input_ids, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    return model

# Example usage with T5
def build_t5_model(input_shape, num_classes, max_len):
    t5_model = TFT5Model.from_pretrained("t5-small")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    t5_output = t5_model(input_ids)[0]
    x = layers.GlobalAveragePooling1D()(t5_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    if num_classes > 1:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    
    model = Model(inputs=input_ids, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    return model

# Define input shape, number of heads, layers, etc.
input_shape = (32, 32, 32, 1)
num_heads = 8
num_layers = 4
d_model = 128
dff = 512
num_classes = 10
max_len = 100

# Example model creation
transformer_model = build_transformer_model(input_shape, num_heads, num_layers, d_model, dff, num_classes, max_len)
transformer_model.summary()

bert_model = build_bert_model(input_shape, num_classes, max_len)
bert_model.summary()

gpt3_model = build_gpt3_model(input_shape, num_classes, max_len)
gpt3_model.summary()

t5_model = build_t5_model(input_shape, num_classes, max_len)
t5_model.summary()

# Example classification tasks
models = [
    {"name": "lstm_chiral", "filter": lambda df: df.assign(chiral=df['chiral_length'].apply(lambda x: 1 if x != 0 else 0)), "label_column": "chiral", "num_classes": 1},
    {"name": "lstm_chiral_01", "filter": lambda df: df[df['chiral_length'].isin([0, 1])].assign(chiral_length_01=df['chiral_length']), "label_column": "chiral_length_01", "num_classes": 1},
    {"name": "lstm_chiral_length", "filter": lambda df: df[df['chiral_length'].isin([0, 1, 2, 3, 4])], "label_column": "chiral_length", "num_classes": 5},
    {"name": "lstm_posneg", "filter": lambda df: df[df['chiral_length'] == 1].assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
    {"name": "lstm_posneg_all", "filter": lambda df: df.assign(rotation0_binary=lambda df: np.where(df['rotation0'] > 0, 1, 0)), "label_column": "rotation0_binary", "num_classes": 1},
    {"name": "lstm_rs", "filter": lambda df: df[df['chiral_length'] == 1].assign(chiral_binary=lambda df: df['chiral0'].apply(lambda x: 1 if x == 'R' else 0)), "label_column": "chiral_binary", "num_classes": 1}
]

def run_model(dataset, labels, input_shape, pooling_type, num_hidden_layers, nodes_per_layer, epochs, model_name, num_classes=1):
    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    # Create generators
    train_generator = tensor_generator(train_data, train_labels, set_batch_size, input_shape)
    test_generator = tensor_generator(test_data, test_labels, set_batch_size, input_shape)

    # Calculate steps per epoch
    steps_per_epoch = len(train_data) // set_batch_size
    validation_steps = len(test_data) // set_batch_size

    # Train the CNN model
    model = build_cnn_model(input_shape, pooling_type, num_hidden_layers, nodes_per_layer, num_classes)
    metrics_callback = MetricsCallback(next(test_generator)[0], test_labels[:set_batch_size])
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=test_generator, validation_steps=validation_steps, verbose=1, callbacks=[metrics_callback])

    # Evaluate the model
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
    input_shape = (resolution, resolution, resolution, 1) 

    # Load dataset
    dataset = pd.read_csv(filename)

    if 'tensor' not in dataset.columns:
        raise KeyError("The dataset does not contain a 'tensor' column.")

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
            nodes_per_layer = 64
        elif model_name == "cnn_chiral_01":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 256
        elif model_name == "cnn_chiral_length":
            pooling_type = 'flatten'
            hidden_layers = 3
            nodes_per_layer = 128
        elif model_name == "cnn_posneg":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 64
        elif model_name == "cnn_posneg_all":
            pooling_type = 'flatten'
            hidden_layers = 2
            nodes_per_layer = 64
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
