"""
predict_images.py - Batch prediction utility for images using a pre-trained Keras model.

This script validates images from a directory, loads a pre-trained model, processes the images,
performs predictions, and saves the results to a CSV file.
"""

import os
import pandas as pd
from PIL import Image
from typing import List, Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.imagenet_utils import smart_resize # type: ignore

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

def predict_from_directory(
    model,
    directory: str,
    target_size: Tuple[int, int]
) -> pd.DataFrame:
    """Predict on valid images using flow_from_dataframe and return results as DataFrame."""

    valid_paths = load_valid_image_paths(directory)
    df = pd.DataFrame({"filename": valid_paths})

    datagen = ImageDataGenerator(preprocessing_function=lambda img: smart_resize(img, size=target_size))
    
    generator = datagen.flow_from_dataframe(
        directory,
        target_size=target_size,
        batch_size=32,
        shuffle=False,
        class_mode=None
    )
    
    preds = model.predict(generator, verbose=1)
    filenames = generator.filenames
    num_classes = preds.shape[1]
    
    df = pd.DataFrame(preds, columns=[f"class_{i}" for i in range(num_classes)])
    df.insert(0, "filename", [os.path.basename(f) for f in filenames])
    return df
