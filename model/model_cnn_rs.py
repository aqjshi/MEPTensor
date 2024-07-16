import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Flatten, Dense, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from argparse import ArgumentParser

print("model_cnn_rs.py")

def f1_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

def build_cnn_model(input_shape):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', f1_m])
    return model

class MetricsCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.best_f1 = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.where(y_pred > 0.5, 1, 0)

        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes, average='weighted')
        recall = recall_score(self.y_test, y_pred_classes, average='weighted')
        f1 = f1_score(self.y_test, y_pred_classes, average='weighted')

        print(f"Epoch {epoch + 1}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch + 1
            self.model.save('best_model.h5')
            print(f"Best model saved at epoch {self.best_epoch} with F1 score: {self.best_f1}")

def process_and_evaluate_model(filename, test_size, input_shape):
    # Load dataset
    dataset = pd.read_csv(filename)

    # Filter dataset for molecules with exactly one chiral center
    dataset = dataset[dataset['chiral_length'] == 1]

    # Ensure tensor data has 729 values
    def parse_tensor(tensor_str):
        values = np.fromstring(tensor_str, sep=' ')
        assert len(values) == 729, f"Tensor does not have 729 values: {len(values)}"
        return values.reshape(9, 9, 9)

    # Convert tensor data
    tensor_data = np.stack(dataset['tensor'].apply(parse_tensor).values)
    tensor_data = tensor_data[..., np.newaxis]  # Add a channel dimension for CNN

    # Labels
    labels = dataset['chiral0'].apply(lambda x: 1 if x == 'R' else 0).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=test_size, random_state=42)

    # Train the CNN model
    model = build_cnn_model(input_shape)
    metrics_callback = MetricsCallback(X_test, y_test)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[metrics_callback])

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    print(f"Final Accuracy: {accuracy}")
    print(f"Final Precision: {precision}")
    print(f"Final Recall: {recall}")
    print(f"Final F1 Score: {f1}")

    return len(dataset), accuracy, precision, recall, f1

def main():
    parser = ArgumentParser()
    parser.add_argument("filename", help="The filename of the dataset")
    parser.add_argument("test_size", type=float, help="The test size for the train-test split")
    args = parser.parse_args()

    filename = args.filename
    test_size = args.test_size
    input_shape = (9, 9, 9, 1)  # Assuming the tensor is 9x9x9 with a single channel

    length, accuracy, precision, recall, f1 = process_and_evaluate_model(filename, test_size, input_shape)
    result = {
        "Length of Filtered Dataset": length,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    print(result)  # Added for debugging

    results_df = pd.DataFrame([result])
    results_df.to_csv("model_rs_results.csv", index=False)

if __name__ == "__main__":
    main()
