
TESTING"
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")

"
Running Tensor.py

python tensor.py [FILENAME] [DENSITY] [PRECISION]
Example

CSV PARSER
python tensor.py xyz.csv 64 2

NPY PARSER


Testing param: python tensor.py qm9_filtered.npy 64 2
Scientific param: python tensor.py qm9_filtered.npy 729 4



#how to visulaize nn
PyTorchViz
visualization package. 38

CITATIONS:

https://irvinelab.uchicago.edu/papers/EI12.pdf
https://www.biorxiv.org/content/10.1101/2022.08.24.505155v1.full.pdf

https://www.nature.com/articles/s41377-020-00367-8

https://www.condmatjclub.org/uploads/2013/06/JCCM_JUNE_2013_03.pdf

https://arxiv.org/pdf/2008.01715
