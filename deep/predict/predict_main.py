"""
predict_images.py - Batch prediction utility for images using a pre-trained Keras model.

This script validates images from a directory, loads a pre-trained model, processes the images,
performs predictions, and saves the results to a CSV file.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

def is_valid_image(file_path: str) -> bool:
    """Check if a file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def load_valid_image_paths(directory: str) -> List[str]:
    """Load and return valid image file paths from a directory."""
    return [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if is_valid_image(os.path.join(directory, fname))
    ]

def preprocess_image(img_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess an image for model prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_batch(model_path: str, image_paths: List[str], target_size: Tuple[int, int]) -> pd.DataFrame:
    """Run predictions on a batch of images."""
    model = keras_load_model(model_path)
    results = []

    for path in image_paths:
        img = preprocess_image(path, target_size)
        pred = model.predict(img)[0]
        results.append((os.path.basename(path), *pred))
        
    num_classes = len(results[0]) - 1
    columns = ["filename"] + [f"class_{i}" for i in range(num_classes)]
    return pd.DataFrame(results, columns=columns)

def predict_images_from_dir(
    model_path: str,
    input_dir: str,
    target_size: Tuple[int, int]
) -> pd.DataFrame:
    """
    Run predictions on all valid images in a directory.

    Args:
        model_path: Path to a Keras .h5 model file.
        input_dir: Directory containing input images.
        target_size: Size (height, width) expected by the model.

    Returns:
        A pandas DataFrame with filenames and class probabilities.
    """
    model = keras_load_model(model_path)
    image_paths = load_valid_image_paths(input_dir)
    results = []

    for path in image_paths:
        img = preprocess_image(path, target_size)
        pred = model.predict(img)[0]
        results.append((os.path.basename(path), *pred))

    num_classes = len(results[0]) - 1
    columns = ["filename"] + [f"class_{i}" for i in range(num_classes)]

    return pd.DataFrame(results, columns=columns)