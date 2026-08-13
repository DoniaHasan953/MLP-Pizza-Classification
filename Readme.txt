MLP Pizza Classification

1. Project Overview
This project is an image classification project using a Multi-Layer Perceptron (MLP) neural network.
The goal is to classify images into two classes:
- Pizza
- Not Pizza

2. Dataset
The project uses the Pizza Not Pizza dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza

The dataset contains:
- 983 Pizza images
- 983 Not Pizza images

Total images: 1966

3. Data Preprocessing
The images were loaded using OpenCV.

The preprocessing steps were:
- Read images using OpenCV.
- Resize all images to 128 x 128.
- Assign labels:
  - Pizza = 1
  - Not Pizza = 0
- Shuffle the data.
- Convert the images and labels into NumPy arrays.
- Normalize pixel values by dividing by 255.
- Split the data into training and testing sets.

Training images: 1572
Testing images: 394

4. Model Architecture
A Multi-Layer Perceptron (MLP) was used for image classification.

Architecture:
- Flatten layer
- Dense layer: 128 neurons, ReLU activation
- Dense layer: 64 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation

5. Model Compilation
Optimizer:
Adam

Loss Function:
Binary Crossentropy

Metric:
Accuracy

6. Model Training
The model was trained for 10 epochs.

Batch size:
32

Validation split:
20% of the training data

7. Model Evaluation
The final test results were:

Test Accuracy: 62.18%
Test Loss: 0.8018

The validation accuracy reached approximately 62.86%.

8. Streamlit Application
A Streamlit application was created to allow the user to upload an image and classify it as Pizza or Not Pizza.

The application:
- Loads the trained MLP model.
- Accepts an image from the user.
- Resizes the image to 128 x 128.
- Normalizes the image.
- Makes a prediction using the trained model.
- Displays the predicted class and confidence.

9. Project Structure

MLP-Pizza-Classification/
│
├── app.py
├── README.txt
├── Req.txt
│
└── src/
    ├── pizza_mlp.ipynb
    └── mlp_model.keras