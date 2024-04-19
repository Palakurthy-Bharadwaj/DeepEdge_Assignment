#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.models import load_model
import os


# <h2>Dataset Loading</h2>

# In[ ]:


# Constants
IMAGE_SIZE = 50

def load_dataset(dataset_dir):
    """
    This function loads images and corresponding labels (x, y coordinates) from a dataset directory.

    Args:
        dataset_dir (str): Path to the directory containing images and label files.

    Returns:
        tuple: A tuple containing two NumPy arrays - images and labels.

    Raises:
        OSError: If the dataset directory is not found.
    """

    images = []
    labels = []

    if not os.path.exists(dataset_dir):
        raise OSError(f"Dataset directory not found: {dataset_dir}")

    # Get list of image files in the dataset directory
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(dataset_dir, image_file)

        # Load image and preprocess
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize image
        images.append(image)

        # Load corresponding label
        label_file = f'label_{os.path.splitext(image_file)[0].split("_")[1]}.txt'
        label_path = os.path.join(dataset_dir, label_file)

        with open(label_path, 'r') as f:
            label = f.read().split()
            label = (int(label[0]), int(label[1]))
        labels.append(label)

    return np.array(images), np.array(labels)


# Load dataset [ function call ]
train_images, train_labels = load_dataset('train_dataset')
test_images, test_labels = load_dataset('test_dataset')


# <h2>Model Creation and Training</h2>

# In[ ]:


# Define model
def create_model():
    """
    This function defines a convolutional neural network architecture for predicting
    (x, y) coordinates of a single white pixel in a grayscale image.

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # Output layer for (x, y) coordinates
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    return model

# Create and train the model
model = create_model()
model.fit(train_images, train_labels, epochs=100, batch_size=32, validation_split=0.2)


# In[ ]:


# Assuming 'model' is your trained Keras model
model.save('my_model.keras')  # Save the model with a .keras extension

