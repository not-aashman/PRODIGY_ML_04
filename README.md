# PRODIGY_ML_04: Hand Gesture Recognition Using CNN

## Task Overview
This repository contains the implementation of a hand gesture recognition model using Convolutional Neural Networks (CNN) to classify 10 different hand gestures. The task was performed as part of the Machine Learning Internship Program at Prodigy InfoTech.

## Dataset
The dataset used for this task is the LeapGestRecog dataset, which consists of 10 categories of hand gestures, including "palm," "fist," "thumb," "index," "ok," and others. Each category contains grayscale images that were resized to 50x50 pixels for uniformity. Due to the large size of the dataset, it cannot be uploaded directly to this repository. You can download the dataset from the following link:

[Download LeapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

Once downloaded, place the dataset in the appropriate directory as outlined in the code.

### Dataset Features:
- **Images**: Grayscale, resized to 50x50 pixels.
- **Labels**: 10 categories of hand gestures.

## Task Description
The task involved building a CNN-based classification model to identify and classify hand gestures into 10 different categories.

### Steps Performed:

#### Data Preprocessing:
- Loaded and resized images to 50x50 pixels in grayscale.
- Split the dataset into training and test sets.
- One-hot encoded the labels to be used in the classification task.
- Normalized the pixel values to a range between 0 and 1 for faster convergence.

#### Model Implementation:
- Implemented a Convolutional Neural Network (CNN) using Keras with layers including Conv2D, MaxPooling, Dropout, and Dense layers.
- Trained the model on the training data using categorical cross-entropy loss and RMSProp optimizer.
- Made predictions on the test data.

#### Model Evaluation:
- Evaluated the model using accuracy and loss metrics.
- Visualized training and validation accuracy and loss over epochs.
- Generated a confusion matrix to assess model performance across different gesture categories.

### Model Architecture:
- 3 Convolutional layers with ReLU activation and MaxPooling.
- Dropout layers for regularization.
- Fully connected Dense layers for classification into 10 gesture categories.

## Model Performance:
- **Accuracy on Test Set**: The model achieved outstanding results with an accuracy of **99.95%**.
- **Loss**: A steady decrease in loss over the training period.
- **Confusion Matrix**: Visualized the model's predictions across categories, demonstrating the model's ability to classify gestures effectively.

## Visualization:
- Plotted training vs. validation loss and accuracy across epochs to monitor performance.
- Displayed a confusion matrix with Seaborn heatmap to visualize classification results.
