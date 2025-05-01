"""
metric_utils.py - Utility functions for visualizing and evaluating models in computer vision tasks.

This module contains several helper functions to assist with common operations like:
- Visualizing augmented images from an image generator.
- Extracting and printing model training metrics, including precision and loss.
- Plotting model training and validation metrics like precision, accuracy, and loss.
- Visualizing confusion matrices for model evaluation.

These functions can be used for quick diagnostics and to ensure that a model is training effectively.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.models import Model #type: ignore

def show_augmented_images(
        generator: ImageDataGenerator, 
        num_images: int = 8
    ) -> None:
    """
    Displays a grid of augmented images from a given image generator.

    This function helps in visualizing how images are being augmented by a generator 
    during training. It shows a specified number of images in a grid format.

    Args:
        generator (ImageDataGenerator): A generator that yields batches of images and their corresponding labels.
        num_images (int): The number of images to display from the generated batch (default is 8).
    """
    images, _ = next(generator)

    actual_num = min(num_images, len(images))

    max_cols = 4
    rows = math.ceil(actual_num / max_cols)
    plt.figure(figsize=(max_cols * 3, rows * 3))

    for i in range(actual_num):
        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_fitted_model_metrics(
        model: Model
        ,is_final: bool = False
    ) -> float:
    """
    Retrieves the best epoch metrics (train/validation loss and precision) from a fitted model.

    This function extracts the model's training history and finds the epoch with the best 
    validation loss. It prints and returns the precision at the best epoch.

    Args:
        model (Model): The fitted Keras model whose history is being analyzed.

    Returns:
        float: The best validation precision achieved during training.
    """
    history = model.history

    # Get best epoch metrics
    if is_final:
        best_epoch = np.argmin(history['loss'])
    
    else:
        best_epoch = np.argmin(history['val_loss'])

    # Loss
    best_train_loss = history['loss'][best_epoch]
    print(f"Train Loss: {best_train_loss}")
    
    if not is_final:
        best_val_loss = history['val_loss'][best_epoch]
        print(f"Validation Loss: {best_val_loss}")
    
    # Precision
    best_train_pre = history['precision'][best_epoch]
    print(f"Train Precision:{best_train_pre}")

    if not is_final:
        best_val_pre = history['val_precision'][best_epoch]
        print(f"Validation Precision: {best_val_pre}")

        return best_val_pre

def plot_metrics(
        model: Model, 
        binary: bool = False
    ) -> None:
    """
    Plots the training and validation metrics such as precision, accuracy, and loss.

    This function generates and displays plots for model training and validation 
    precision, accuracy, and loss across epochs. It is designed to help visualize 
    how well a model is training.

    Args:
        model (Model): The Keras model with training history.
        binary (bool): Whether the model is a binary classifier. Default is False (multi-class).
    """
    history = model.history

    # Defining the variables (multiclass)
    precision = history['precision']
    val_precision = history['val_precision']
    loss = history['loss']
    val_loss = history['val_loss']

    if not binary:
        accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']

    epochs = range(1, len(precision) + 1)

    # Precision Plot
    plt.plot(epochs, precision, 'bo', label='Training Precision')
    plt.plot(epochs, val_precision, 'b', label='Validation Precision')
    plt.title("Training and Validation Precision")
    plt.legend()
    plt.figure()

    if not binary:
        # Accuracy Plot
        plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.figure()

    # Loss Plot
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.legend()

    # Show the plots
    plt.show()

def plot_confusion_matrix(
        y_true: np.ndarray
        , y_pred: np.ndarray
        , labels: list
    ) -> None:

    """
    Plots a confusion matrix to evaluate the classification performance of the model.

    This function computes the confusion matrix based on the true and predicted labels 
    and visualizes it as a heatmap. It helps in understanding the performance of a 
    classification model in terms of false positives, false negatives, and correctly 
    predicted labels.

    Args:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.
        labels (list): The list of class labels.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=[*labels],
        yticklabels=[*labels]
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
