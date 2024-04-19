#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tabulate import tabulate
from math import sqrt


# <h2>Image Preprocessing</h2>

# In[22]:


# Constants for image size
IMAGE_SIZE = 50

def preprocess_test_image(image_path):
    """
    This function preprocesses a single test image by loading, converting to grayscale,
    normalizing, and adding a batch dimension.

    Args:
        image_path (str): Path to the test image file.

    Returns:
        np.ndarray: The preprocessed test image as a NumPy array.
    """
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# In[23]:


# Path to the test dataset directory
test_dataset_dir = 'test_dataset'

# Path to your saved Keras model file
model_path = 'my_model.keras' 
model = tf.keras.models.load_model(model_path) 


# <h2>Batch Prediction & Evaluation of the Test dataset</h2>

# In[24]:


def evaluate_model(model_path, test_dataset_dir):
    """
    This function loads a trained model, iterates over test images in a directory,
    predicts the (x, y) coordinates, compares them to ground truth labels, and calculates
    the prediction error (e.g., Root Mean Squared Error).

    Args:
        model_path (str): Path to the saved Keras model file.
        test_dataset_dir (str): Path to the directory containing test images and labels.
    """
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # Iterate over test images in the test dataset directory
    test_image_files = [f for f in os.listdir(test_dataset_dir) if f.endswith('.png')]
    
    # Create lists to store ground truth and predicted coordinates
    ground_truth = []
    predictions = []
    x_errors = []
    y_errors = []
    
    for test_image_file in test_image_files:
        # Construct the full path to the test image
        test_image_path = os.path.join(test_dataset_dir, test_image_file)
    
        # Preprocess the test image
        test_image = preprocess_test_image(test_image_path)
    
        # Predict coordinates using the trained model
        predicted_coords = model.predict(test_image)[0]
        predicted_x, predicted_y = int(round(predicted_coords[0])), int(round(predicted_coords[1]))
    
        # Load ground truth label (coordinates) from the corresponding label file
        label_file = f'label_{os.path.splitext(test_image_file)[0].split("_")[1]}.txt'
        label_path = os.path.join(test_dataset_dir, label_file)
        with open(label_path, 'r') as f:
            true_x, true_y = map(int, f.read().split())
    
        # Append ground truth and predicted coordinates to the lists
        ground_truth.append((true_x, true_y))
        predictions.append((predicted_x, predicted_y))
    
        # Calculate and store errors
        x_error = abs(predicted_x - true_x)
        y_error = abs(predicted_y - true_y)
        x_errors.append(x_error)
        y_errors.append(y_error)
    
    # Create a table using the tabulate library
    table = tabulate(
        [["Image", "Ground Truth", "Predicted"]] +
        [
            [f"Image {i+1}", str(ground_truth[i]), str(predictions[i])]
            for i in range(len(ground_truth))
        ],
        headers="firstrow",
        tablefmt="grid"
    )
    
    print(table)
    
    # Calculate and print evaluation metrics
    mae_x = sum(x_errors) / len(x_errors)
    mae_y = sum(y_errors) / len(y_errors)
    rmse_x = sqrt(sum(x ** 2 for x in x_errors) / len(x_errors))
    rmse_y = sqrt(sum(y ** 2 for y in y_errors) / len(y_errors))
    
    print(f"\nMean Absolute Error (MAE):")
    print(f"X-coordinate: {mae_x}")
    print(f"Y-coordinate: {mae_y}")
    
    print(f"\nRoot Mean Squared Error (RMSE):")
    print(f"X-coordinate: {rmse_x}")
    print(f"Y-coordinate: {rmse_y}")


# In[26]:


evaluate_model(model_path, test_dataset_dir)
print("\nModel evaluation complete.")


# <h2>Single Image Prediction</h2>

# In[27]:


#Give a path to a foreign image with a "255" in only cell with 0 in every cell outside the dataset to verify

TEST_IMAGE_PATH = "C:\\Users\\palak\\Deepedge\\test_dataset\\image_916.png"  # Path to your test image file

# Preprocess the single test image
test_image = preprocess_test_image(TEST_IMAGE_PATH)

# Predict coordinates using the trained model
predicted_coords = model.predict(test_image)[0]
predicted_x, predicted_y = int(round(predicted_coords[0])), int(round(predicted_coords[1]))

# Load ground truth label (coordinates) from the corresponding label file
test_image_filename = os.path.basename(TEST_IMAGE_PATH)
label_file = f'label_{os.path.splitext(test_image_filename)[0].split("_")[1]}.txt'
label_path = os.path.join(os.path.dirname(TEST_IMAGE_PATH), label_file)
with open(label_path, 'r') as f:
    true_x, true_y = map(int, f.read().split())

# Print predicted and ground truth coordinates
print(f"Predicted Coordinates: ({predicted_x}, {predicted_y})")
print(f"Ground Truth Coordinates: ({true_x}, {true_y})")

