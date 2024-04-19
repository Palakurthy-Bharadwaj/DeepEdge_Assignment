#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import os
import cv2
from random import shuffle
from sklearn.model_selection import train_test_split


# In[9]:


# Constants for image size and pixel count
IMAGE_SIZE = 50
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# <h2>Main dataset generation</h2>

# In[10]:


def generate_data(output_dir):
    """
    This function generates unique grayscale images (50x50) with a single white pixel
    at a random location and saves them along with their corresponding coordinates (x, y)
    in separate text files.

    Args:
        output_dir (str): Path to the directory where images and labels will be saved.
    """

    pixel_positions = [(x, y) for y in range(IMAGE_SIZE) for x in range(IMAGE_SIZE)]
    shuffle(pixel_positions)

    for index, (x, y) in enumerate(pixel_positions):
        # Create image with single white pixel at (x, y)
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        image[y, x] = 255

        # Save image with filename format 'image_{index}.png'
        image_path = os.path.join(output_dir, f'image_{index}.png')
        cv2.imwrite(image_path, image)

        # Save label (coordinates) in a text file named 'label_{index}.txt'
        label_path = os.path.join(output_dir, f'label_{index}.txt')
        with open(label_path, 'w') as f:
            f.write(f'{x} {y}')



OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)
generate_data(OUTPUT_DIR)
print(f"Data generation complete. Images and labels saved in {OUTPUT_DIR}")


# <h2>Train & Test dataset creation</h2>

# In[11]:


# Split data into train and test sets
image_paths = [os.path.join(OUTPUT_DIR, f'image_{i}.png') for i in range(NUM_PIXELS)]
label_paths = [os.path.join(OUTPUT_DIR, f'label_{i}.txt') for i in range(NUM_PIXELS)]
train_image_paths, test_image_paths, train_label_paths, test_label_paths = train_test_split(
    image_paths, label_paths, test_size=0.01, random_state=42
)

# Move train images and labels to train directory
train_dir = 'train_dataset'
os.makedirs(train_dir, exist_ok=True)
for image_path, label_path in zip(train_image_paths, train_label_paths):
    image_name = os.path.basename(image_path)
    label_name = os.path.basename(label_path)
    os.rename(image_path, os.path.join(train_dir, image_name))
    os.rename(label_path, os.path.join(train_dir, label_name))

# Move test images and labels to test directory
test_dir = 'test_dataset'
os.makedirs(test_dir, exist_ok=True)
for image_path, label_path in zip(test_image_paths, test_label_paths):
    image_name = os.path.basename(image_path)
    label_name = os.path.basename(label_path)
    os.rename(image_path, os.path.join(test_dir, image_name))
    os.rename(label_path, os.path.join(test_dir, label_name))

print("Train and test datasets are created")

