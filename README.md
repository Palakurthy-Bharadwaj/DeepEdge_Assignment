**Introduction**

This project implements a deep learning approach using a Convolutional Neural Network (CNN) to predict the (x, y) coordinates of a single white pixel (value 255) in a 50x50 grayscale image.

**Dependencies**

* Python 3
* TensorFlow 
* NumPy
* OpenCV



**Rationale behind the dataset choice:**

- As mentioned in the document, the image size is fixed, i.e., 50x50. So irrespective of the choice of dataset, the maximum number of distinct images that can be generated that satisfy our task is 2500.
- Randomly assigning the white pixel to each image ensures that the model learns to identify pixel positions based on image features rather than memorizing specific coordinates.
- This approach promotes the development of a robust model that can accurately predict pixel locations, regardless of their placement within the image.
- Exposing the model to a wide range of pixel positions through random assignment fosters generalization, preventing bias towards specific patterns.
- This particular approach results in a controlled way of creating the dataset while not sacrificing randomness.




**Data Generation (optional - already included in the repo)**

The script `data_generation.py` generates a synthetic dataset of 50x50 grayscale images with a single white pixel at a random location. Corresponding labels (x, y) coordinates are saved in separate text files.



**Model Training (optional - saved model is provided in the repo)**

The script `model_training.py` performs the following steps:

1. Loads the generated dataset (images and labels).
2. Defines a CNN architecture suitable for image classification.
3. Compiles the model with an optimizer (Adam) and a loss function (Mean Squared Error).
4. Trains the model on the training data with a validation split for monitoring performance.
5. Saves the model.



**Model Evaluation**

The script `model_evaluation.py` performs the following steps:

1. Loads the trained model.
2. Iterates over test images in the test dataset.
3. Preprocesses each test image (grayscale conversion, normalization).
4. Predicts the (x, y) coordinates using the model.
5. Loads the ground truth coordinates from the corresponding label file.
6. Calculates and prints the prediction error (Mean Absolute Error, Root Mean Squared Error).



**Running the Scripts locally:**

1. Clone this repository.
2. Install the required dependencies (refer to Installation section).
3. Navigate to the project directory in your terminal.
4. Run the data generation script (optional, if you want to generate a new dataset):
   ```bash
   python data_generation.py
   ```
5. Train the model:
   ```bash
   python model_training.py
   ```
6. Evaluate the model on the test set:
   ```bash
   python model_evaluation.py
   ```
