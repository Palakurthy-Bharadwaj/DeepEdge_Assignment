Introduction

This project implements a deep learning approach using a Convolutional Neural Network (CNN) to predict the (x, y) coordinates of a single white pixel (value 255) in a 50x50 grayscale image.

Dependencies

Python 3.x
TensorFlow (tested with version X.X - replace with your version)
NumPy
OpenCV

Installation

Install Python from https://www.python.org/downloads/.
Create a virtual environment (recommended) to isolate project dependencies:
Bash
python -m venv venv
source venv/bin/activate  # Activate the virtual environment (Windows: venv\Scripts\activate)
Use code with caution.
Install required libraries within the virtual environment:
Bash
pip install tensorflow numpy opencv-python
Use code with caution.
Data Generation

The script data_generation.py generates a synthetic dataset of 50x50 grayscale images with a single white pixel at a random location. Corresponding labels (x, y) coordinates are saved in separate text files.

Model Training

The script model_training.py performs the following steps:

Loads the generated dataset (images and labels).
Defines a CNN architecture suitable for image classification.
Compiles the model with an optimizer (Adam) and a loss function (Mean Squared Error).
Trains the model on the training data with a validation split for monitoring performance. (Optional: consider using early stopping to prevent overfitting)
Model Evaluation

The script model_evaluation.py performs the following steps:

Loads the trained model.
Iterates over test images in the test dataset.
Preprocesses each test image (grayscale conversion, normalization).
Predicts the (x, y) coordinates using the model.
Loads the ground truth coordinates from the corresponding label file.
Calculates and prints the prediction error (e.g., Mean Absolute Error, Root Mean Squared Error).
Saving and Loading the Model

The script model_saving_loading.py demonstrates how to save the trained model using model.save() and load it back for future predictions using tf.keras.models.load_model().

Running the Scripts

Locally:

Clone this repository or download the compressed file.
Install the required dependencies (refer to Installation section).
Navigate to the project directory in your terminal.
Run the data generation script (optional, if you want to generate a new dataset):
Bash
python data_generation.py
Use code with caution.
Train the model:
Bash
python model_training.py
Use code with caution.
Evaluate the model on the test set:
Bash
python model_evaluation.py
